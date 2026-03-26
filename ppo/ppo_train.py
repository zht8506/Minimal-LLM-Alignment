"""
Simple PPO (Proximal Policy Optimization) Training for LLM.
"""

import argparse
import json
import os
import time
import sys
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

# Ensure local dataset module can be imported no matter where this script is launched.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from gsm8k_dataset import GSM8KJsonDataset, gsm8k_collate_fn
from gsm8k_reward import compute_gsm8k_reward_batch


def masked_mean(tensor: torch.Tensor, mask: torch.BoolTensor, dim: int = -1) -> torch.Tensor:
    """Compute mean over masked positions."""
    return (tensor * mask).sum(dim) / mask.sum(dim).clamp(min=1)


def compute_log_probs(
    logits: torch.Tensor,           # (B, S, V), V is the vocab size
    input_ids: torch.Tensor,        # (B, S)
    action_mask: torch.BoolTensor,  # (B, S)  True = generated token, False = prompt token
) -> torch.Tensor:
    """Per-token log probability for generated tokens."""
    
    # shift: logits[t] predicts token[t+1]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)   # (B, S-1, V)
    token_ids = input_ids[:, 1:]                             # (B, S-1)
    per_token_lp = log_probs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1)  # (B, S-1)

    # action_mask is defined on response tokens (starting at prompt_len)
    return per_token_lp * action_mask  # zero out prompt positions


def compute_approx_kl(
    log_probs: torch.Tensor,           # (B, S-1)
    ref_log_probs: torch.Tensor,       # (B, S-1)
    action_mask: torch.BoolTensor,     # (B, S-1)
) -> torch.Tensor:
    """
    k1 estimator (https://github.com/verl-project/verl/blob/main/verl/trainer/ppo/core_algos.py#L2152)
    """
    kl = log_probs - ref_log_probs            # (B, S-1)

    return masked_mean(kl, action_mask)       # (B,)


def compute_gae(
    values: torch.Tensor,          # (B, S-1)  — value at each token position
    rewards: torch.Tensor,         # (B, S-1)  — dense reward (KL-penalised)
    action_mask: torch.BoolTensor, # (B, S-1)
    gamma: float = 1.0,
    lambd: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation.
    Returns advantages (B, S-1) and returns (B, S-1).
    """
    # mask out non-response positions
    values  = values  * action_mask
    rewards = rewards * action_mask

    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    lastgae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        next_val = values[:, t + 1] if t < T - 1 else torch.zeros(B, device=values.device)
        delta = rewards[:, t] + gamma * next_val - values[:, t] # Advantage Estimation
        lastgae = delta + gamma * lambd * lastgae  # Generalized Advantage Estimation (GAE)
        advantages[:, t] = lastgae

    returns = advantages + values

    return advantages.detach(), returns.detach()


class PPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # validate micro/mini batch sizes
        assert args.ppo_mini_batch_size % args.ppo_micro_batch_size_per_gpu == 0, (
            f"ppo_mini_batch_size ({args.ppo_mini_batch_size}) must be divisible by "
            f"ppo_micro_batch_size_per_gpu ({args.ppo_micro_batch_size_per_gpu})"
        )
        self.grad_accum_steps = args.ppo_mini_batch_size // args.ppo_micro_batch_size_per_gpu

        # ── models ──
        print(f"Loading model from {args.model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # left-pad for generation

        self.actor = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            trust_remote_code=True,
        ).to(self.device)

        # Critic: lightweight value head on top of a llm backbone
        self.critic_backbone = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            trust_remote_code=True,
        ).to(self.device)
        hidden_size = self.critic_backbone.config.hidden_size
        critic_dtype = next(self.critic_backbone.parameters()).dtype
        self.value_head = torch.nn.Linear(hidden_size, 1, bias=False).to(device=self.device, dtype=critic_dtype)
        torch.nn.init.zeros_(self.value_head.weight)

        # Reference model: frozen copy of the initial actor
        self.ref_model = deepcopy(self.actor)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        self.actor_optim = torch.optim.AdamW(
            self.actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.weight_decay,
        )
        critic_params = list(self.critic_backbone.parameters()) + list(self.value_head.parameters())
        self.critic_optim = torch.optim.AdamW(
            critic_params,
            lr=args.critic_lr,
            weight_decay=args.weight_decay,
        )

        self.dataset = GSM8KJsonDataset(args.train_json, self.tokenizer)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.train_batch_size,   # #prompts per rollout
            shuffle=True,
            collate_fn=gsm8k_collate_fn,
            drop_last=False,
        )

        self.eval_dataloader = None
        if args.eval_steps > 0:
            if not args.eval_json:
                raise ValueError("When eval_steps > 0, --eval_json must be set.")
            eval_dataset = GSM8KJsonDataset(args.eval_json, self.tokenizer)
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=gsm8k_collate_fn,
                drop_last=False,
            )
            print(
                f"Eval enabled: every {args.eval_steps} step(s), "
                f"{len(eval_dataset)} samples, batch_size={args.eval_batch_size}"
            )

        # estimate total optimizer steps for scheduler:
        # steps_per_epoch = ceil(dataset / train_batch_size)
        # each step does (train_batch_size // ppo_mini_batch_size) mini-batch updates
        steps_per_epoch = len(self.dataloader)
        mini_updates_per_step = max(args.train_batch_size // args.ppo_mini_batch_size, 1)
        total_optimizer_steps = args.num_train_epochs * steps_per_epoch * mini_updates_per_step

        self.actor_scheduler  = get_linear_schedule_with_warmup(
            self.actor_optim,  num_warmup_steps=args.warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
        self.critic_scheduler = get_linear_schedule_with_warmup(
            self.critic_optim, num_warmup_steps=args.warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        self.log_file = open(os.path.join(args.output_dir, "train_log.jsonl"), "a")

        print(
            f"Dataset size: {len(self.dataset)}  |  "
            f"Steps per epoch: {steps_per_epoch}  |  "
            f"grad_accum_steps: {self.grad_accum_steps}  |  "
            f"Estimated total global steps: {args.num_train_epochs * steps_per_epoch }"
        )
