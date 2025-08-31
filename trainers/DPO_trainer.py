import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from omegaconf import DictConfig
from typing import Optional, Dict, List, Union, Tuple

from .base_trainer import BaseTrainer
from .loss import preference_loss, _get_batch_logps
from utils import (
    all_gather_if_needed,
    pad_to_length,
    rank0_print,
)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    
    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class DPOTrainer(BaseTrainer):
    """DPO trainer that handles preference learning with chosen/rejected pairs."""
    
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, cache_dir: Optional[str] = None):
        """Initialize DPO trainer."""
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size, cache_dir)

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        all_logits = model(concatenated_batch['concatenated_input_ids'], attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        return chosen_logps, rejected_logps

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the preference loss (e.g., dpo, cdpo, ipo, etc) and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        policy_chosen_logps, policy_rejected_logps = self.concatenated_forward(self.policy, batch)
        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

        loss_kwargs = {'loss_type': loss_config.name, 'beta': loss_config.beta, 'reference_free': loss_config.reference_free,
                        'label_smoothing': loss_config.label_smoothing}

        losses, chosen_rewards, rejected_rewards = preference_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, **loss_kwargs)

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
        rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
        reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

        metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
        metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

        policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics
