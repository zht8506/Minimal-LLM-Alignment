import torch
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional, Dict, List, Union

from .base_trainer import BaseTrainer
from .loss import _get_batch_logps
from utils import all_gather_if_needed


class SFTTrainer(BaseTrainer):
    """SFT trainer that handles supervised fine-tuning."""
    
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, cache_dir: Optional[str] = None):
        """Initialize SFT trainer."""
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size, cache_dir)

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        policy_chosen_logits = self.policy(batch['input_ids'], attention_mask=batch['attention_mask']).logits.to(torch.float32)
        policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['labels'], average_log_prob=False)

        losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        # metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics
