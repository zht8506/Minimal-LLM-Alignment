import os
import argparse
import json
import socket
import multiprocessing as mp
import resource
from typing import Optional, Set
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
import wandb
from omegaconf import DictConfig, OmegaConf

from utils import get_open_port, disable_dropout, init_distributed, get_output_dir_structure
from trainers.trainer_factory import get_trainer_class


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    
    # ------------------- initialize wandb configuration -------------------------------#
    # disables wandb if config.debug=True
    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = config.cache_dir
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=config.cache_dir,
            name=config.exp_name,
        )

    # ------------------- obtain trainers and begin train -------------------------------#
    TrainerClass = get_trainer_class(config)
    trainer = TrainerClass(policy, config, config.seed, config.runs_dir, reference_model=reference_model, rank=rank, world_size=world_size, cache_dir=config.cache_dir)
    
    print(f'Creating trainer on process {rank} with world size {world_size}')

    trainer.train()
    trainer.save()


def main():

    # --------------- Parsing parameters from config files -------------------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to the YAML configuration file")
    parser.add_argument("--overrides", nargs="*", help="Config overrides in the form key=value")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_file)
    
    if args.overrides:
        cli_config = OmegaConf.from_cli(args.overrides)
        config = OmegaConf.merge(config, cli_config)

    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    
    # ------------- Create new directory structure and save config ------------------------#
    # Create new directory structure
    base_dir, cache_dir, runs_dir = get_output_dir_structure(config.output_dirs)
    config.cache_dir = cache_dir
    config.runs_dir = runs_dir

    os.environ['XDG_CACHE_HOME'] = config.cache_dir # Cache temporary file address

    print(OmegaConf.to_yaml(config))

    # make dir and save config
    config_path = os.path.join(config.runs_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{base_dir}')
    print(f'Cache directory: {cache_dir}')
    print(f'Runs directory: {runs_dir}')
    print('=' * 80)

    # ------------------------------- Building model -----------------------------------------#
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=config.cache_dir, low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)

    reference_model = None
    if config.loss.name in {'dpo', 'ipo'}:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=config.cache_dir, low_cpu_mem_usage=True, torch_dtype=reference_model_dtype,** model_kwargs)
        disable_dropout(reference_model)

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name in {'dpo', 'ipo'}:
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')
    
    # ------------------------------------ Training Process------------------------------------------#
    print('starting single-process worker')
    worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()
