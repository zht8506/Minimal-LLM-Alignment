"""
Simple GSPO (Group Sequence Policy Optimization) Training for LLM.
"""

import argparse
import json
import os
import time
import sys
from pathlib import Path
from copy import deepcopy
from typing import List, Tuple, Optional

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

class GSPOTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
        self.tokenizer.padding_side = "left"

        self.actor = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            trust_remote_code=True,
        ).to(self.device)

        # Reference model: frozen copy of the initial actor (for KL penalty)
        self.ref_model = deepcopy(self.actor)
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

        self.actor_optim = torch.optim.AdamW(
            self.actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.weight_decay,
        )

        # ── data ──
        self.dataset = GSM8KJsonDataset(args.train_json, self.tokenizer)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=args.train_batch_size,
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

        # scheduler
        steps_per_epoch = len(self.dataloader)
        total_responses_per_step = args.train_batch_size * args.num_answers_per_question
        mini_updates_per_step = max(total_responses_per_step // args.ppo_mini_batch_size, 1)
        total_optimizer_steps = args.num_train_epochs * steps_per_epoch * mini_updates_per_step

        self.actor_scheduler = get_linear_schedule_with_warmup(
            self.actor_optim, num_warmup_steps=args.warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        self.log_file = open(os.path.join(args.output_dir, "train_log.jsonl"), "a")

        print(
            f"Dataset size: {len(self.dataset)}  |  "
            f"Steps per epoch: {steps_per_epoch}  |  "
            f"train_batch_size: {args.train_batch_size}  |  "
            f"num_answers_per_question (G): {args.num_answers_per_question}  |  "
            f"total responses per step: {total_responses_per_step}  |  "
            f"grad_accum_steps: {self.grad_accum_steps}  |  "
            f"Estimated total global steps: {args.num_train_epochs * steps_per_epoch}"
        )

    # ------------------------------------------------------------------
    # Step 1 · Rollout: generate G responses per prompt
    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        prompts: List[str],
        rollout_n: Optional[int] = None,
        *,
        do_sample: bool = True,
        set_train_after: bool = True,
    ) -> dict:
        """
        Generate *num_answers* responses for each of the B prompts.
        Returns tensors of shape (B*G, S) where G = num_answers.
        """
        args = self.args
        G = args.num_answers_per_question if rollout_n is None else rollout_n

        expanded_prompts = []
        for p in prompts:
            expanded_prompts.extend([p] * G)

        prompt_tokens = self.tokenizer(
            expanded_prompts,
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

        sequences = output                       # (B*G, S)
        attention_mask = (sequences != self.tokenizer.pad_token_id).long()

        action_mask = torch.zeros(sequences.shape[0], sequences.shape[1] - 1,
                                  dtype=torch.bool, device=self.device)
        action_mask[:, prompt_len - 1:] = True
        action_mask = action_mask & attention_mask[:, 1:].bool()

        # group_ids: maps each of the B*G responses back to its prompt index
        group_ids = torch.arange(len(prompts), device=self.device).repeat_interleave(G)

        return {
            "sequences":      sequences,         # (B*G, S)
            "attention_mask": attention_mask,    # (B*G, S)
            "action_mask":    action_mask,       # (B*G, S-1)
            "prompt_len":     prompt_len,
            "group_ids":      group_ids,         # (B*G, )
        }

    # ------------------------------------------------------------------
    # Step 2 · Make experience: group-relative advantage (no critic)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def make_experience(
        self,
        rollout_data: dict,
        rewards: torch.Tensor,   # (N,)  scalar reward per response, N = B*G
    ) -> dict:
        args = self.args
        sequences      = rollout_data["sequences"]
        attention_mask = rollout_data["attention_mask"]
        action_mask    = rollout_data["action_mask"]
        group_ids      = rollout_data["group_ids"]

        autocast_ctx = torch.autocast(device_type=self.device.type,
                                      dtype=torch.bfloat16, enabled=args.bf16)

        # ── actor (old policy) log-probs ──
        with autocast_ctx:
            actor_out = self.actor(input_ids=sequences, attention_mask=attention_mask)
        old_log_probs = compute_log_probs(actor_out.logits, sequences, action_mask)

        # ── reference model log-probs ──
        with autocast_ctx:
            ref_out = self.ref_model(input_ids=sequences, attention_mask=attention_mask)
        ref_log_probs = compute_log_probs(ref_out.logits, sequences, action_mask)

        kl_per_seq = compute_approx_kl(old_log_probs, ref_log_probs, action_mask)  # (N,)

        # ── GRPO advantage: Group-relative advantage normalization ──
        # For each group (same prompt), normalize: adv_i = (r_i - mean) / (std + eps)
        advantages = torch.zeros_like(rewards)
        for gid in group_ids.unique():
            mask = group_ids == gid
            group_rewards = rewards[mask]
            mean_r = group_rewards.mean()
            std_r = group_rewards.std()
            advantages[mask] = (group_rewards - mean_r) / (std_r + 1e-4)

        return {
            "sequences":      sequences,
            "attention_mask": attention_mask,
            "action_mask":    action_mask,
            "old_log_probs":  old_log_probs.detach(),
            "ref_log_probs":  ref_log_probs.detach(),
            "advantages":     advantages.detach(),       # (N,)  per-response scalar
            "kl":             kl_per_seq.mean().item(),
        }

    # ------------------------------------------------------------------
    # Step 3 · GSPO update: group-wise advantage calculation + KL penalty
    # ------------------------------------------------------------------
    def gspo_update(self, experience: dict) -> dict:
        """
        Split into mini-batches of ppo_mini_batch_size.
        Each mini-batch is further split into micro-batches of
        ppo_micro_batch_size_per_gpu for gradient accumulation.
        grad_accum_steps = ppo_mini_batch_size // ppo_micro_batch_size_per_gpu
        """
        args = self.args
        N = experience["sequences"].shape[0]

        total_actor_loss = 0.0
        total_clip_ratio = 0.0
        total_entropy    = 0.0
        n_updates = 0

        autocast_ctx = torch.autocast(device_type=self.device.type,
                                      dtype=torch.bfloat16, enabled=args.bf16)

        indices = torch.randperm(N)

        for mini_start in range(0, N, args.ppo_mini_batch_size):
            mini_idx = indices[mini_start: mini_start + args.ppo_mini_batch_size]
            if len(mini_idx) == 0:
                continue

            actual_accum = max(
                (len(mini_idx) + args.ppo_micro_batch_size_per_gpu - 1)
                // args.ppo_micro_batch_size_per_gpu,
                1,
            )

            self.actor_optim.zero_grad(set_to_none=True)

            mini_actor_loss = 0.0
            mini_clip_ratio = 0.0
            mini_entropy    = 0.0

            for micro_start in range(0, len(mini_idx), args.ppo_micro_batch_size_per_gpu):
                micro_idx = mini_idx[micro_start: micro_start + args.ppo_micro_batch_size_per_gpu]

                sequences      = experience["sequences"][micro_idx]
                attention_mask = experience["attention_mask"][micro_idx]
                action_mask    = experience["action_mask"][micro_idx]
                old_log_probs  = experience["old_log_probs"][micro_idx]
                ref_log_probs  = experience["ref_log_probs"][micro_idx]
                advantages     = experience["advantages"][micro_idx]       # (micro_B,)

                # ── new actor forward ──
                with autocast_ctx:
                    actor_out = self.actor(input_ids=sequences, attention_mask=attention_mask)
                new_log_probs = compute_log_probs(actor_out.logits, sequences, action_mask)

                # ──────────────────────────── GSPO loss begin ──────────────────────────
                # ── log ratio: log(π_θ / π_θ_old) ──
                log_ratio = new_log_probs - old_log_probs

                seq_lengths = torch.sum(action_mask, dim=-1).clamp(min=1)
                # sequence-level importance ratio, s_i(θ) in paper
                log_seq_importance_ratio = torch.sum(log_ratio * action_mask, dim=-1) / seq_lengths

                # GSPO-token importance ratio, s_i,t(θ) in paper
                # when advantage is sequence-level, the s_i,t(θ) = s_i(θ)
                log_seq_importance_ratio_token = new_log_probs - new_log_probs.detach() + log_seq_importance_ratio.detach().unsqueeze(-1)
                log_seq_importance_ratio_token = torch.clamp(log_seq_importance_ratio_token, max=10.0)  # clamp for numerical stability

                # finaly exp() to remove log
                seq_importance_ratio = torch.exp(log_seq_importance_ratio_token)

                # broadcast per-response advantage → per-token
                token_advantages = advantages.unsqueeze(1).expand_as(seq_importance_ratio) * action_mask

                surr1 = seq_importance_ratio * token_advantages
                surr2 = seq_importance_ratio.clamp(1 - args.clip_eps_low, 1 + args.clip_eps_high) * token_advantages
                actor_loss = -masked_mean(torch.min(surr1, surr2), action_mask).mean()

                # ── KL penalty (per-token, with ref model) ──
                # ── Unlike PPO, KL in the loss for GRPO ──
                kl_penalty = masked_mean(new_log_probs - ref_log_probs, action_mask).mean()

                # ── entropy loss ──
                logits = actor_out.logits[:, :-1, :]
                probs  = logits.softmax(dim=-1)
                entropy = -(probs * probs.clamp(min=1e-8).log()).sum(-1)
                entropy_loss = -masked_mean(entropy, action_mask).mean()

                clip_ratio = masked_mean((surr2 < surr1).float(), action_mask).mean() # for logging

                loss = (actor_loss + args.kl_coef * kl_penalty + args.entropy_coef * entropy_loss) / actual_accum
                loss.backward()
                # ──────────────────────────── GSPO loss begin ──────────────────────────

                mini_actor_loss += actor_loss.item()
                mini_clip_ratio += clip_ratio.item()
                mini_entropy    += masked_mean(entropy, action_mask).mean().item()

            # ── optimizer step after full mini-batch ──
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
            self.actor_optim.step()
            self.actor_scheduler.step()

            total_actor_loss += mini_actor_loss / actual_accum
            total_clip_ratio += mini_clip_ratio / actual_accum
            total_entropy    += mini_entropy    / actual_accum
            n_updates += 1

        k = max(n_updates, 1)
        return {
            "actor_loss":  total_actor_loss / k,
            "clip_ratio":  total_clip_ratio / k,
            "entropy":     total_entropy    / k,
            "actor_lr":    self.actor_scheduler.get_last_lr()[0],
        }


    # ------------------------------------------------------------------
    # Evaluation (greedy, 1 response per prompt)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_eval(self) -> dict:
        args = self.args
        assert self.eval_dataloader is not None

        reward_sum  = 0.0
        success_sum = 0.0
        n_total     = 0

        for batch in self.eval_dataloader:
            prompts = batch["prompts"]
            targets = batch["targets"]
            bsz = len(prompts)

            rollout_data = self.rollout(
                prompts, rollout_n=1, do_sample=False, set_train_after=False
            )
            sequences  = rollout_data["sequences"]
            prompt_len = rollout_data["prompt_len"]

            response_ids = sequences[:, prompt_len:]
            responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            rewards_list = compute_gsm8k_reward_batch(
                responses=responses,
                ground_truths=targets,
                method="flexible",
                format_score=0.0,
                score=1.0,
            )
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
            reward_sum  += rewards.sum().item()
            success_sum += (rewards == 1.0).float().sum().item()
            n_total     += bsz

        self.actor.train()
        mean_reward  = reward_sum  / max(n_total, 1)
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
        G = args.num_answers_per_question

        print(f"Start GRPO training for {args.num_train_epochs} epoch(s).")

        for epoch in range(1, args.num_train_epochs + 1):

            for batch in self.dataloader:
                step_t0 = time.time()
                global_step += 1

                prompts = batch["prompts"]
                targets = batch["targets"]
                B = len(prompts)

                # ── Step 1: Rollout — G responses per prompt ──
                rollout_data = self.rollout(prompts)
                sequences  = rollout_data["sequences"]     # (B*G, S)
                prompt_len = rollout_data["prompt_len"]

                # ── decode & reward all B*G responses ──
                response_ids = sequences[:, prompt_len:]
                responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

                expanded_targets = []
                for t in targets:
                    expanded_targets.extend([t] * G)

                rewards_list = compute_gsm8k_reward_batch(
                    responses=responses,
                    ground_truths=expanded_targets,
                    method="flexible",
                    format_score=0.0,
                    score=1.0,
                )
                rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)

                # ── Step 2: Make experience (group-relative advantages) ──
                experience = self.make_experience(rollout_data, rewards)

                # ── Step 3: GRPO update ──
                update_stats = self.gspo_update(experience)

                step_duration = time.time() - step_t0

                # ── logging ──
                mean_reward  = rewards.mean().item()
                success_rate = (rewards == 1.0).float().mean().item()
                log_payload = {
                    "epoch":          epoch,
                    "global_step":    global_step,
                    "mean_reward":    round(mean_reward,  4),
                    "success_rate":   round(success_rate, 4),
                    "kl":             round(experience["kl"], 6),
                    "actor_loss":     round(update_stats["actor_loss"],  6),
                    "clip_ratio":     round(update_stats["clip_ratio"],  4),
                    "entropy":        round(update_stats["entropy"],     4),
                    "actor_lr":       update_stats["actor_lr"],
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
                        "eval_mean_reward":  round(eval_stats["eval_mean_reward"], 4),
                        "eval_success_rate": round(eval_stats["eval_success_rate"], 4),
                        "eval_num_samples":  eval_stats["eval_num_samples"],
                        "eval_duration_sec": round(time.time() - eval_t0, 2),
                    }
                    print(eval_payload)
                    self.log_file.write(json.dumps(eval_payload) + "\n")
                    self.log_file.flush()

                # ── save checkpoint (no critic to save) ──
                if global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"epoch{epoch}-step{global_step}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    self.actor.save_pretrained(ckpt_dir)
                    self.tokenizer.save_pretrained(ckpt_dir)
                    print(f"[epoch {epoch} | step {global_step}] Checkpoint saved to {ckpt_dir}")

            ckpt_dir = os.path.join(args.output_dir, f"epoch{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            self.actor.save_pretrained(ckpt_dir)
            self.tokenizer.save_pretrained(ckpt_dir)
            print(f"[Epoch {epoch}] Checkpoint saved to {ckpt_dir}")

        self.log_file.close()
        print("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Simple GRPO training for LLM")
    # paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model.")
    parser.add_argument("--train_json", type=str, required=True, help="Path to training json file.")
    parser.add_argument("--output_dir", type=str, default="output_dir/grpo")
    # generation
    parser.add_argument("--max_prompt_len", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    # GRPO hyper-params
    parser.add_argument("--train_batch_size", type=int, default=8,
                        help="Number of unique prompts sampled per rollout step.")
    parser.add_argument("--num_answers_per_question", type=int, default=4,
                        help="Number of responses generated per prompt (group size G). "
                             "Total rollout size = train_batch_size * G.")
    parser.add_argument("--clip_eps_low", type=float, default=0.0003, help="PPO-clip epsilon.")
    parser.add_argument("--clip_eps_high", type=float, default=0.0004, help="PPO-clip epsilon.")
    parser.add_argument("--kl_coef", type=float, default=0.05, help="KL penalty coefficient (ref model).")
    parser.add_argument("--entropy_coef", type=float, default=0, help="Entropy loss coefficient. Default 0 for GRPO.")
    # optimizer
    parser.add_argument("--actor_lr", type=float, default=1e-6)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    # training schedule
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--ppo_mini_batch_size", type=int, default=16,
                        help="Mini-batch size for policy gradient updates. "
                             "Must be divisible by ppo_micro_batch_size_per_gpu.")
    parser.add_argument("--ppo_micro_batch_size_per_gpu", type=int, default=4,
                        help="Micro-batch size per forward/backward pass (gradient accumulation).")
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
    trainer = GSPOTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
