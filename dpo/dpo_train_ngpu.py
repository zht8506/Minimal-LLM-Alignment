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
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from contextlib import nullcontext
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_PATH = Path(__file__).parent.parent / "datasets"
sys.path.insert(0, str(DATA_PATH))

from dpo_dataset import JsonDPODataset, DPODataCollator


def _get_batch_logps(
    logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False
) -> torch.FloatTensor:
    assert logits.shape[:-1] == labels.shape
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != -100
    labels[labels == -100] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    return (per_token_logps * loss_mask).sum(-1)
