from typing import Optional, Dict, Any
from omegaconf import DictConfig

from trainers.DPO_trainers import DPOTrainer
from trainers.SFT_trainers import SFTTrainer


def get_trainer_class(config: DictConfig) -> type:
    """
    Get the appropriate trainer class based on configuration.
    
    Args:
        config: Configuration object containing trainer and loss settings
        
    Returns:
        The trainer class to use
        
    Raises:
        ValueError: If the configuration is invalid or unsupported
    """
    # Determine the training type (DPO or SFT)
    loss_name = config.loss.name.lower()
    
    # Determine the trainer type (Basic or FSDP)
    trainer_type = config.trainer
    
    if loss_name in ['dpo', 'ipo']:
        # DPO training
        if trainer_type == 'FSDPTrainer':
            return FSDPDPOTrainer
        elif trainer_type == 'BasicTrainer':
            return DPOTrainer
        else:
            raise ValueError(f"Unsupported trainer type '{trainer_type}' for DPO training. Use 'Basic' or 'FSDP'.")
    
    elif loss_name == 'sft':
        # SFT training
        if trainer_type == 'FSDPTrainer':
            return FSDP_SFTTrainer
        elif trainer_type == 'BasicTrainer':
            return SFTTrainer
        else:
            raise ValueError(f"Unsupported trainer type '{trainer_type}' for SFT training. Use 'Basic' or 'FSDP'.")
    
    else:
        raise ValueError(f"Unsupported loss type '{loss_name}'. Supported types: 'dpo', 'ipo', 'sft'")
