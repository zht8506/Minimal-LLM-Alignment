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
    kl_estimator: str = "k1"
) -> torch.Tensor:
    """k1 KL estimator."""
    log_ratio = log_probs - ref_log_probs            # (B, S-1)
    
    if kl_estimator == "k1":
        pass  # log_ratio is already p - q
    elif kl_estimator == "k2":
        # Non-negative KL approximation: (p - q)^2 / 2
        # http://joschu.net/blog/kl-approx.html
        # Approximately equivalent to one-step KL penalty with k1
        # used in https://arxiv.org/pdf/2310.10505.
        log_ratio = log_ratio**2 / 2.0
    elif kl_estimator == "k3":
        # Non-negative KL approximation: exp(q - p) - 1 - (q - p)
        # http://joschu.net/blog/kl-approx.html
        log_ratio = (-log_ratio).exp() - 1 + log_ratio
    else:
        raise ValueError(f"Unknown kl_estimator: {kl_estimator}")

    return masked_mean(log_ratio.clamp(min=-10, max=10), action_mask)       # (B,)

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

    # ------------------------------------------------------------------
    # Step 1 · Rollout: generate responses with the current actor
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        prompts: List[str],
        *,
        do_sample: bool = True,
        set_train_after: bool = True,
    ) -> dict:
        """
        Generate one response per prompt.
        prompts: list of B prompt strings (B = train_batch_size)
        """
        args = self.args

        prompt_tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_len,
        ).to(self.device)

        prompt_len = prompt_tokens["input_ids"].shape[1]

        self.actor.eval()
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=args.bf16):
            output = self.actor.generate(**prompt_tokens, **gen_kwargs)
        if set_train_after:
            self.actor.train()

        # sequences: (B, prompt_len + response_len)
        sequences = output  # (B, S)
        attention_mask = (sequences != self.tokenizer.pad_token_id).long()

        # action_mask: True for generated (response) tokens, shape (B, S-1)
        action_mask = torch.zeros(sequences.shape[0], sequences.shape[1] - 1,
                                  dtype=torch.bool, device=self.device)
        action_mask[:, prompt_len - 1:] = True
        # also mask out padding in response
        action_mask = action_mask & attention_mask[:, 1:].bool()

        return {
            "sequences":      sequences,        # (B, S)
            "attention_mask": attention_mask,   # (B, S)
            "action_mask":    action_mask,      # (B, S-1)
            "prompt_len":     prompt_len,
        }

    # ------------------------------------------------------------------
    # Step 2 · Make experience: forward all models, compute advantages
    # ------------------------------------------------------------------
    @torch.no_grad()
    def make_experience(
        self,
        rollout_data: dict,
        rewards: torch.Tensor,   # (N,)  scalar reward per response
    ) -> dict:
        args = self.args
        sequences      = rollout_data["sequences"]
        attention_mask = rollout_data["attention_mask"]
        action_mask    = rollout_data["action_mask"]

        autocast_ctx = torch.autocast(device_type=self.device.type,
                                      dtype=torch.bfloat16, enabled=args.bf16)

        with autocast_ctx:
            actor_out = self.actor(
                input_ids=sequences,
                attention_mask=attention_mask,
            )
        old_log_probs = compute_log_probs(actor_out.logits, sequences, action_mask)  # (N, S-1)

        with autocast_ctx:
            ref_out = self.ref_model(
                input_ids=sequences,
                attention_mask=attention_mask,
            )
        ref_log_probs = compute_log_probs(ref_out.logits, sequences, action_mask)    # (N, S-1)

        with autocast_ctx:
            critic_out = self.critic_backbone(
                input_ids=sequences,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden = critic_out.hidden_states[-1][:, :-1, :]    # (N, S-1, H)
        hidden = hidden.to(self.value_head.weight.dtype)
        values = self.value_head(hidden).squeeze(-1)        # (N, S-1)

        # ── KL per sequence (mean over response tokens) ──
        kl_per_seq = compute_approx_kl(old_log_probs, ref_log_probs, action_mask)  # (N,)

        # ── shaped reward: rewards_kl = rewards - kl_coef * KL ──
        # Put the scalar reward at the EOS position, KL every token.
        kl_dense = (old_log_probs - ref_log_probs) * action_mask            # (N, S-1)
        rewards_kl = -args.kl_coef * kl_dense                               # (N, S-1)
        # add the scalar reward at the last valid token
        for i in range(sequences.shape[0]):
            # find the idx of last non-zero value
            last_idx = (action_mask[i].nonzero(as_tuple=False)[-1].item() if action_mask[i].any() else 0)
            # add reward to the last token
            rewards_kl[i, last_idx] += rewards[i]

        # ── GAE advantages & returns ──
        advantages, returns = compute_gae(values, rewards_kl, action_mask, gamma=args.gamma, lambd=args.lambd)

        # ── normalise advantages (over the whole batch) ──
        valid_adv = advantages[action_mask]
        adv_mean, adv_std = valid_adv.mean(), valid_adv.std().clamp(min=1e-8)
        advantages = (advantages - adv_mean) / adv_std

        return {
            "sequences":      sequences,
            "attention_mask": attention_mask,
            "action_mask":    action_mask,
            "old_log_probs":  old_log_probs.detach(),
            "values":         values.detach(),
            "advantages":     advantages.detach(),
            "returns":        returns.detach(),
            "kl":             kl_per_seq.mean().item(),
        }

    # ------------------------------------------------------------------
    # Step 3 · PPO update: batch_size -> mini_batch_size -> micro_batch_size (grad accumulation)
    # ------------------------------------------------------------------
    def ppo_update(self, experience: dict) -> dict:
        """
        Split into mini-batches of ppo_mini_batch_size.
        Each mini-batch is further split into micro-batches of
        ppo_micro_batch_size_per_gpu for gradient accumulation.
        grad_accum_steps = ppo_mini_batch_size // ppo_micro_batch_size_per_gpu
        """
        args = self.args
        N = experience["sequences"].shape[0] # train_batch_size

        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_clip_ratio  = 0.0
        n_updates = 0

        autocast_ctx = torch.autocast(device_type=self.device.type,
                                      dtype=torch.bfloat16, enabled=args.bf16)

        indices = torch.randperm(N)

        for mini_start in range(0, N, args.ppo_mini_batch_size):
            mini_idx = indices[mini_start: mini_start + args.ppo_mini_batch_size]
            if len(mini_idx) == 0:
                continue

            # accumulation steps
            actual_accum = max(
                (len(mini_idx) + args.ppo_micro_batch_size_per_gpu - 1)
                // args.ppo_micro_batch_size_per_gpu,
                1,
            )

            self.actor_optim.zero_grad(set_to_none=True)
            self.critic_optim.zero_grad(set_to_none=True)

            mini_actor_loss  = 0.0
            mini_critic_loss = 0.0
            mini_clip_ratio  = 0.0

            # ── micro-batch loop (gradient accumulation) ──
            for micro_start in range(0, len(mini_idx), args.ppo_micro_batch_size_per_gpu):
                micro_idx = mini_idx[micro_start: micro_start + args.ppo_micro_batch_size_per_gpu]

                sequences      = experience["sequences"][micro_idx]
                attention_mask = experience["attention_mask"][micro_idx]
                action_mask    = experience["action_mask"][micro_idx]
                old_log_probs  = experience["old_log_probs"][micro_idx]
                old_values     = experience["values"][micro_idx]
                advantages     = experience["advantages"][micro_idx]
                returns        = experience["returns"][micro_idx]

                # ── Actor loss with PPO-clip ──
                with autocast_ctx:
                    actor_out = self.actor(
                        input_ids=sequences,
                        attention_mask=attention_mask,
                    )
                new_log_probs = compute_log_probs(actor_out.logits, sequences, action_mask)

                # ──────────────────────────── PPO loss begin ──────────────────────────
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp() # importance sampling

                surr1 = ratio * advantages
                surr2 = ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps) * advantages
                actor_loss = -masked_mean(torch.min(surr1, surr2), action_mask).mean()

                # ── entropy loss ──
                logits = actor_out.logits[:, :-1, :]
                probs  = logits.softmax(dim=-1)
                entropy = -(probs * probs.clamp(min=1e-8).log()).sum(-1) # entropy = p * logp
                entropy_loss = -masked_mean(entropy, action_mask).mean()

                clip_ratio = masked_mean((surr2 < surr1).float(), action_mask).mean()

                # ── Critic loss (clipped value loss) ──
                with autocast_ctx:
                    critic_out = self.critic_backbone(
                        input_ids=sequences,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                hidden = critic_out.hidden_states[-1][:, :-1, :]
                hidden = hidden.to(self.value_head.weight.dtype)
                new_values = self.value_head(hidden).squeeze(-1)

                values_clipped = old_values + (new_values - old_values).clamp(
                    -args.value_clip_eps, args.value_clip_eps
                )
                vf_loss1 = (new_values     - returns) ** 2
                vf_loss2 = (values_clipped - returns) ** 2
                critic_loss = 0.5 * masked_mean(torch.max(vf_loss1, vf_loss2), action_mask).mean()

                # scale loss by 1/actual_accum to average gradients
                actor_total_loss  = (actor_loss + args.entropy_coef * entropy_loss) / actual_accum
                critic_total_loss = (args.vf_coef * critic_loss) / actual_accum

                actor_total_loss.backward()
                critic_total_loss.backward()
                # ──────────────────────────── PPO loss end ──────────────────────────
                

                mini_actor_loss  += actor_loss.item()
                mini_critic_loss += critic_loss.item()
                mini_clip_ratio  += clip_ratio.item()

            # ── optimizer step after full mini-batch accumulated ──
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
            self.actor_optim.step()
            self.actor_scheduler.step()

            torch.nn.utils.clip_grad_norm_(
                list(self.critic_backbone.parameters()) + list(self.value_head.parameters()),
                args.max_grad_norm,
            )
            self.critic_optim.step()
            self.critic_scheduler.step()

            total_actor_loss  += mini_actor_loss  / actual_accum
            total_critic_loss += mini_critic_loss / actual_accum
            total_clip_ratio  += mini_clip_ratio  / actual_accum
            n_updates += 1

        k = max(n_updates, 1)
        return {
            "actor_loss":   total_actor_loss  / k,
            "critic_loss":  total_critic_loss / k,
            "clip_ratio":   total_clip_ratio  / k,
            "actor_lr":     self.actor_scheduler.get_last_lr()[0],
        }

    # ------------------------------------------------------------------
    # Evaluation (greedy generation, no PPO update)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_eval(self) -> dict:
        args = self.args
        assert self.eval_dataloader is not None

        reward_sum = 0.0
        success_sum = 0.0
        n_total = 0

        for batch in self.eval_dataloader:
            prompts = batch["prompts"]
            targets = batch["targets"]
            bsz = len(prompts)

            rollout_data = self.rollout(prompts, do_sample=False, set_train_after=False)
            sequences = rollout_data["sequences"]
            prompt_len = rollout_data["prompt_len"]

            response_ids = sequences[:, prompt_len:]
            responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            rewards_list = compute_gsm8k_reward_batch(
                responses=responses,
                ground_truths=targets,
                method="strict",
                format_score=0.0,
                score=1.0,
            )
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
            reward_sum += rewards.sum().item()
            success_sum += (rewards == 1.0).float().sum().item()
            n_total += bsz

        self.actor.train()
        mean_reward = reward_sum / max(n_total, 1)
        success_rate = success_sum / max(n_total, 1)
        return {
            "eval_mean_reward":  mean_reward,
            "eval_success_rate": success_rate,
            "eval_num_samples":  n_total,
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(self):
        args = self.args
        global_step = 0

        print(f"Start PPO training for {args.num_train_epochs} epoch(s).")

        for epoch in range(1, args.num_train_epochs + 1):

            for batch in self.dataloader:
                step_t0 = time.time()
                global_step += 1

                prompts = batch["prompts"]
                targets = batch["targets"]
                B = len(prompts)

                # ── Step 1: Rollout and Reward──
                rollout_data = self.rollout(prompts)
                sequences  = rollout_data["sequences"]
                prompt_len = rollout_data["prompt_len"]

                # ── decode responses & compute rewards ──
                response_ids = sequences[:, prompt_len:]
                responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

                rewards_list = compute_gsm8k_reward_batch(
                    responses=responses,
                    ground_truths=targets,
                    method="flexible",
                    format_score=0.0,
                    score=1.0,
                )
                rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)  # (B,)

                # ── Step 2: make experience ──
                experience = self.make_experience(rollout_data, rewards)

                # ── Step 3: PPO update ──
                update_stats = self.ppo_update(experience)

                step_duration = time.time() - step_t0

                # ── logging ──
                mean_reward  = rewards.mean().item()
                success_rate = (rewards == 1.0).float().mean().item()
                log_payload = {
                    "epoch":        epoch,
                    "global_step":  global_step,
                    "mean_reward":  round(mean_reward,  4),
                    "success_rate": round(success_rate, 4),
                    "kl":           round(experience["kl"], 6),
                    "actor_loss":   round(update_stats["actor_loss"],  6),
                    "critic_loss":  round(update_stats["critic_loss"], 6),
                    "clip_ratio":   round(update_stats["clip_ratio"],  4),
                    "actor_lr":     update_stats["actor_lr"],
                    "step_duration":  round(step_duration, 2),
                }

                if global_step % args.logging_steps == 0:
                    print(log_payload)
                    self.log_file.write(json.dumps(log_payload) + "\n")
                    self.log_file.flush()

                if self.eval_dataloader is not None and global_step % args.eval_steps == 0:
                    eval_t0 = time.time()
                    eval_stats = self.run_eval()
                    eval_payload = {
                        "type": "eval",
                        "epoch": epoch,
                        "global_step": global_step,
                        "eval_mean_reward": round(eval_stats["eval_mean_reward"], 4),
                        "eval_success_rate": round(eval_stats["eval_success_rate"], 4),
                        "eval_num_samples": eval_stats["eval_num_samples"],
                        "eval_duration_sec": round(time.time() - eval_t0, 2),
                    }
                    print(eval_payload)
                    self.log_file.write(json.dumps(eval_payload) + "\n")
                    self.log_file.flush()

                # ── save checkpoint ──
                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"epoch{epoch}-step{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    self.actor.save_pretrained(ckpt_dir)
                    self.tokenizer.save_pretrained(ckpt_dir)
                    critic_dir = os.path.join(ckpt_dir, "critic")
                    os.makedirs(critic_dir, exist_ok=True)
                    self.critic_backbone.save_pretrained(critic_dir)
                    torch.save(self.value_head.state_dict(),
                               os.path.join(critic_dir, "value_head.pt"))
                    print(f"[epoch {epoch} | step {global_step}] Checkpoint saved to {ckpt_dir}")

            ckpt_dir = os.path.join(args.output_dir, f"epoch{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            self.actor.save_pretrained(ckpt_dir)
            self.tokenizer.save_pretrained(ckpt_dir)
            critic_dir = os.path.join(ckpt_dir, "critic")
            os.makedirs(critic_dir, exist_ok=True)
            self.critic_backbone.save_pretrained(critic_dir)
            torch.save(self.value_head.state_dict(),
                       os.path.join(critic_dir, "value_head.pt"))
            print(f"[Epoch {epoch}] Checkpoint saved to {ckpt_dir}")

        self.log_file.close()
        print("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple PPO training for LLM")
    # paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model.")
    parser.add_argument("--train_json", type=str, required=True, help="Path to training json file.")
    parser.add_argument("--output_dir", type=str, default="output_dir/ppo")
    # generation
    parser.add_argument("--max_prompt_len", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    # PPO hyper-params
    parser.add_argument("--gamma", type=float, default=1.0, help="Discount factor.")
    parser.add_argument("--lambd", type=float, default=0.95, help="GAE lambda.")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="PPO clip epsilon.")
    parser.add_argument("--value_clip_eps", type=float, default=0.2, help="Value function clip epsilon.")
    parser.add_argument("--kl_coef", type=float, default=0.05, help="KL penalty coefficient.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy loss coefficient.")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value loss coefficient.")
    # optimizer
    parser.add_argument("--actor_lr", type=float, default=1e-6)
    parser.add_argument("--critic_lr", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # training schedule
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=256,
                        help="Number of prompts sampled per PPO iteration.")
    parser.add_argument("--ppo_mini_batch_size", type=int, default=64,
                        help="Number of trajectories per policy gradient update (mini-batch). "
                             "Must be divisible by ppo_micro_batch_size_per_gpu.")
    parser.add_argument("--ppo_micro_batch_size_per_gpu", type=int, default=8,
                        help="Number of trajectories per forward/backward pass on one GPU. "
                             "Controls gradient accumulation: "
                             "grad_accum = ppo_mini_batch_size // ppo_micro_batch_size_per_gpu.")
    parser.add_argument("--warmup_steps", type=int, default=10)
    # misc
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16.")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N global steps.")
    parser.add_argument("--save_steps", type=int, default=100000,
                        help="Save checkpoint every N global steps (also saves at epoch end).")
    # eval
    parser.add_argument("--eval_steps", type=int, default=0,
                        help="Run eval every N global steps (0 = disabled). When > 0, --eval_json is required.")
    parser.add_argument("--eval_json", type=str, default=None,
                        help="Path to eval JSON (same format as train). Used when eval_steps > 0.")
    parser.add_argument("--eval_batch_size", type=int, default=None,
                        help="Batch size for eval DataLoader. Defaults to train_batch_size when omitted.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.train_batch_size
    return args


def main():
    args = parse_args()
    trainer = PPOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()

