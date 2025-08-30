import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from omegaconf import DictConfig

from dataset.dataset_selector import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    rank0_print,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
from typing import Optional, Dict
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer:
    """Base class for all trainers with common functionality."""
    
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1, cache_dir: Optional[str] = None):
        """Initialize base trainer with common setup."""
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.cache_dir = cache_dir
        
        # Initialize logging
        self.log_file = os.path.join(self.run_dir, 'training_log.json')
        self.log_data = []
        # Create log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=self.cache_dir)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Common data iterator kwargs
        data_iterator_kwargs = dict(
            datasets=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            seed=seed,
            silent=rank != 0,
            cache_dir=self.cache_dir
        )

        self.policy = policy
        self.reference_model = reference_model

        self.train_iterator = get_batch_iterator(
                **data_iterator_kwargs, 
                loss_type = config.loss.name,
                split='train', 
                n_epochs=config.n_epochs, 
                n_examples=config.n_examples, 
                batch_size=config.total_batch_size
            )
        rank0_print(f'Loaded train data iterator')

        self.eval_iterator = get_batch_iterator(
                **data_iterator_kwargs, 
                loss_type = config.loss.name,
                split='test', 
                n_examples=config.n_eval_examples, 
                batch_size=config.eval_batch_size
            )
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

        # Initialize TensorBoard writer based on config
        self.tensorboard_enabled = getattr(config, 'report_to_tensorboard', False)
        if self.tensorboard_enabled and rank == 0:  # Only initialize on rank 0 to avoid conflicts
            tensorboard_dir = os.path.join(self.run_dir, 'tensorboard')
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
            rank0_print(f'TensorBoard logging enabled. Logs will be saved to: {tensorboard_dir}')
        else:
            self.tensorboard_writer = None
            if rank == 0 and not self.tensorboard_enabled:
                rank0_print('TensorBoard logging disabled.')

    def train(self):

        # ------------------------ basic configuration -------------------------------#

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))
    
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo'}:
            self.reference_model.eval()

        self.step_counter = 0
        self.batch_counter = 0
        last_log = None

        self.print_log_step = getattr(self.config, 'print_log_step', 10)
        self.eval_every_steps = getattr(self.config, 'eval_every_steps', 1e9)
        self.save_model_step = getattr(self.config, 'save_model_step', self.eval_every_steps)
        self.save_optimizer = getattr(self.config, 'save_optimizer', False)
        
        # Calculate total steps
        self.total_steps = self._calculate_total_steps()

        # ------------------------ Begin training process -------------------------------#
        for batch in self.train_iterator:

            # ------------------------ Begin evaluation -------------------------------#
            if self.step_counter % self.eval_every_steps == 0 and (self.step_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.step_counter} train steps')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)

                for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                    for k, v in eval_metrics.items():
                        all_eval_metrics[k].extend(v)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.step_counter}: {formatted_dict(mean_eval_metrics)}')
                
                # Save evaluation log to JSON
                eval_log_entry = {
                    'step': self.step_counter,
                    'timestamp': time.time(),
                    'type': 'eval',
                    'metrics': mean_eval_metrics
                }
                self.save_log_to_json(eval_log_entry)

                # Log to TensorBoard/wandb only if enabled
                if self.tensorboard_enabled and self.rank == 0:
                    self.log_to_tensorboard(mean_eval_metrics, self.step_counter, 'eval')
                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.step_counter)

           
            # ------------------------ Begin training -------------------------------#
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.total_batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.step_counter += 1

            # Log based on print_log_step
            if self.step_counter % self.print_log_step == 0:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/steps'] = self.step_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                
                # Log to TensorBoard only if enabled
                if self.tensorboard_enabled and self.rank == 0:
                    self.log_to_tensorboard(mean_train_metrics, self.step_counter, 'train')
                
                # Add timestamp to log entry
                log_entry = {
                    'step': self.step_counter,
                    'timestamp': time.time(),
                    'metrics': mean_train_metrics
                }
                
                rank0_print(f'train stats after {self.step_counter}/{self.total_steps} steps: {formatted_dict(mean_train_metrics)}')
                
                # Save to JSON log file
                self.save_log_to_json(log_entry)

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.step_counter)

                last_log = time.time()
            
            # Save model based on save_model_step
            if self.step_counter % self.save_model_step == 0 and self.step_counter > 0:
                if self.config.debug:
                    rank0_print('skipping save in debug mode')
                else:
                    output_dir = os.path.join(self.run_dir, f'step-{self.step_counter}')
                    rank0_print(f'creating checkpoint to write to {output_dir}...')
                    self.save(output_dir, None)  # No metrics for training saves

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def _calculate_total_steps(self) -> int:
        """Calculate total number of training steps based on dataset size, epochs, and batch size."""
        # For iterator-based datasets, estimate based on configuration
        if self.config.n_examples is not None:
            total_steps = self.config.n_examples // self.config.total_batch_size
            if self.config.n_examples % self.config.total_batch_size != 0:
                total_steps += 1
        elif self.config.n_epochs is not None:
            # Estimate dataset size - this is a rough estimate
            # In practice, you might want to implement a way to get actual dataset size
            estimated_dataset_size = 10000  # Adjust based on your typical dataset size TODO put value to replace estimated_dataset_size
            total_examples = estimated_dataset_size * self.config.n_epochs
            total_steps = total_examples // self.config.total_batch_size
        else:
            # Default fallback
            total_steps = 1000
        
        rank0_print(f"Calculated total steps: {total_steps}")
        return total_steps
    
    def save_log_to_json(self, log_entry: Dict):
        """Save a log entry to the JSON log file."""
        if self.rank == 0:  # Only save on rank 0
            try:
                # Read existing log data
                if os.path.exists(self.log_file):
                    with open(self.log_file, 'r') as f:
                        self.log_data = json.load(f)
                else:
                    self.log_data = []
                
                # Add new log entry
                self.log_data.append(log_entry)
                
                # Write back to file
                with open(self.log_file, 'w') as f:
                    json.dump(self.log_data, f, indent=2)
            except Exception as e:
                rank0_print(f"Warning: Failed to save log to JSON: {e}")

    def log_to_tensorboard(self, metrics: Dict, step: int, prefix: str = ''):
        """Log metrics to TensorBoard."""
        if not self.tensorboard_enabled or self.tensorboard_writer is None:
            return
            
        for key, value in metrics.items():
            # Handle different types of values
            if isinstance(value, list):
                # For list values, log the mean
                if len(value) > 0:
                    mean_value = sum(value) / len(value)
                    tag = f"{prefix}/{key}" if prefix else key
                    self.tensorboard_writer.add_scalar(tag, mean_value, step)
            elif isinstance(value, (int, float)):
                # For scalar values
                tag = f"{prefix}/{key}" if prefix else key
                self.tensorboard_writer.add_scalar(tag, value, step)
            # Skip other types (strings, etc.)

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.step_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        # Save optimizer state dict only if configured to do so
        if self.save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
            self.write_state_dict(self.step_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
            del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.step_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)

        # Close TensorBoard writer when saving final checkpoint (only if enabled)
        if output_dir is None and self.tensorboard_enabled and self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            rank0_print("TensorBoard writer closed.")

    def __del__(self):
        """Cleanup TensorBoard writer when trainer is destroyed."""
        if (hasattr(self, 'tensorboard_enabled') and self.tensorboard_enabled and 
            hasattr(self, 'tensorboard_writer') and self.tensorboard_writer is not None):
            self.tensorboard_writer.close()
