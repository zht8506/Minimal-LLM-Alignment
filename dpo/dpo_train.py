import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local dataset module can be imported no matter where this script is launched.
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dpo_dataset import JsonDPODataset, DPODataCollator


def _get_batch_logps(
    logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
) -> torch.FloatTensor:
    """
    Calculate the GT label log probabilities of each sample.
    """
    assert logits.shape[:-1] == labels.shape
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100
    labels[labels == -100] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    return (per_token_logps * loss_mask).sum(-1)

def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    beta: float,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def move_batch_to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


@dataclass
class TrainState:
    global_step: int = -1
    optimizer_step: int = -1


class BaseTrainer:
    def __init__(self, model, tokenizer, train_loader: DataLoader, args):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.args = args
        self.state = TrainState()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        updates_step_per_epoch = max(1, ceil(len(self.train_loader) / args.gradient_accumulation_steps))

        self.total_optimizer_steps = updates_step_per_epoch * args.num_train_epochs

        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=args.warmup_start_factor,
            end_factor=1.0,
            total_iters=args.warmup_ratio * self.total_optimizer_steps,
        )

        self.log_file = os.path.join(args.output_dir, "train_log.jsonl")
        os.makedirs(args.output_dir, exist_ok=True)

    def get_batch_metrics(self, batch, train: bool = True):
        raise NotImplementedError

    def _save_log(self, payload):
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _save_checkpoint(self, tag: str):
        ckpt_dir = os.path.join(self.args.output_dir, tag)
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        torch.save(self.optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
        torch.save(self.scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))

    def train(self):
        scaler_enabled = torch.cuda.is_available() and self.args.bf16
        autocast_dtype = torch.bfloat16 if scaler_enabled else None
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.args.num_train_epochs):
            start_epoch_time = time.time()

            for step, batch in enumerate(self.train_loader, start=1):
                self.state.global_step += 1
                batch = move_batch_to_device(batch, self.device)

                if torch.cuda.is_available():
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=scaler_enabled):
                        loss, metrics = self.get_batch_metrics(batch, train=True)
                else:
                    with nullcontext():
                        loss, metrics = self.get_batch_metrics(batch, train=True)

                loss_to_backward = loss / self.args.gradient_accumulation_steps
                loss_to_backward.backward()

                if self.state.global_step % self.args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.state.optimizer_step += 1

                    if self.state.optimizer_step % self.args.logging_steps == 0:
                        log_payload = {
                            "epoch": epoch,
                            "optimizer_step": self.state.optimizer_step,
                            "total_step": self.total_optimizer_steps,
                            "loss": float(loss),
                            "lr": self.scheduler.get_last_lr()[0],
                            "grad_norm": float(grad_norm),
                            **metrics,
                        }
                        print(log_payload)
                        self._save_log(log_payload)

                    if self.state.optimizer_step % self.args.save_steps == 0:
                        self._save_checkpoint(f"step-{self.state.optimizer_step}")
            
            epoch_time = time.time() - start_epoch_time
            print({"epoch": epoch, "epoch_seconds": epoch_time})

        self._save_checkpoint("final")


class DPOTrainer(BaseTrainer):
    def __init__(self, policy, ref_model, tokenizer, train_loader: DataLoader, args):
        super().__init__(policy, tokenizer, train_loader, args)
        self.ref_model = ref_model
        self.ref_model.to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def separate_forward(self, model: torch.nn.Module, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:

        # Forward for chosen
        chosen_logits = model(
            batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        ).logits.to(torch.float32)
        chosen_logps = _get_batch_logps(
            chosen_logits, batch["chosen_labels"], average_log_prob=False
        )

        # Forward for rejected
        rejected_logits = model(
            batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        ).logits.to(torch.float32)
        rejected_logps = _get_batch_logps(
            rejected_logits, batch["rejected_labels"], average_log_prob=False
        )

        return chosen_logps, rejected_logps

    def get_batch_metrics(self, batch, train: bool = True):
        policy_chosen_logps, policy_rejected_logps = self.separate_forward(self.model, batch)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps = self.separate_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            beta=self.args.beta
        )

        reward_acc = (chosen_rewards > rejected_rewards).float().mean().item()
        metrics = {
            "reward_accuracy": reward_acc,
            "chosen_reward_mean": chosen_rewards.mean().item(),
            "rejected_reward_mean": rejected_rewards.mean().item(),
        }
        return losses.mean(), metrics


def parse_args():
    parser = argparse.ArgumentParser(description="DPO trainer.")
    # paths
    parser.add_argument("--model_path", type=str, required=True, help="Path to Model file.")
    parser.add_argument("--train_json", type=str, required=True, help="Path to DPO json file.")
    parser.add_argument("--output_dir", type=str, default="./outputs/simple-dpo")
    # hyper-params
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    # optimizer
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_start_factor", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # misc
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=1000000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 autocast on CUDA.")
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading policy, reference, and tokenizer...")
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = JsonDPODataset(
        json_path=args.train_json,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    data_collator = DPODataCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=data_collator,
    )

    trainer = DPOTrainer(
        policy=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        args=args,
    )
    trainer.train()


if __name__ == "__main__":
    main()

