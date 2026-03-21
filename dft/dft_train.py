import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure local dataset module can be imported no matter where this script is launched.
DATA_PATH = Path(__file__).parent.parent / "datasets"
sys.path.insert(0, str(DATA_PATH))

from sft_dataset import JsonSFTDataset, SFTDataCollator

def compute_dft_loss(logits: torch.Tensor, labels: torch.Tensor, dft_alpha: float = 0.0) -> torch.Tensor:
    # Standard causal LM shift: predict token[t] from hidden state at t-1.
    shift_logits = logits[:, :-1, :].contiguous()   # [bs, sq_len, vocab_size]
    shift_labels = labels[:, 1:].contiguous()       # [bs, sq_len]
    vocab_size = shift_logits.size(-1)

    shift_logits_flat = shift_logits.view(-1, vocab_size)  # [bs * sq_len, vocab_size]
    shift_labels_flat = shift_labels.view(-1)              # [bs*seq_len]

    # calculate standard sft loss for each token
    loss_flat = F.cross_entropy(shift_logits_flat, shift_labels_flat, ignore_index=-100, reduction="none") # [bs*seq_len]

    # dft loss
    with torch.no_grad():
        probs = F.softmax(shift_logits_flat, dim=-1) # [bs * sq_len, vocab_size]

        # obtain the prob of gt token
        valid_mask = shift_labels_flat != -100
        # use gather_labels to avoid the -100 in valid_mask
        gather_labels = shift_labels_flat.clone() # [bs * sq_len]
        gather_labels[~valid_mask] = 0
        
        p_correct = probs.gather(1, gather_labels.unsqueeze(-1)).squeeze(-1)
        p_correct = p_correct * valid_mask.float()
        
        dft_weight = p_correct * dft_alpha + (1 - dft_alpha)
        
    weighted_loss = loss_flat * dft_weight

    valid_token_num = valid_mask.sum().clamp(min=1)
    dft_loss = weighted_loss.sum() / valid_token_num

    return dft_loss

def move_batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}
