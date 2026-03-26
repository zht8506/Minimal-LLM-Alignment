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
