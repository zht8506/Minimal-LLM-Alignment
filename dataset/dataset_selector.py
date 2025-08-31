from dataset.preference_dataset import get_dpo_batch_iterator
from dataset.sft_dataset import get_sft_batch_iterator
from utils import rank0_print

def get_batch_iterator(loss_type, *args, **kwargs):
    """return different dataset iterator based on loss_type"""

    if loss_type == "sft":
        rank0_print('Using SFT dataset iterator')
        return get_sft_batch_iterator(*args, **kwargs)

    elif loss_type in ["dpo", "ipo"]:
        rank0_print('Using preference dataset iterator')
        return get_dpo_batch_iterator(*args, **kwargs)

    else:
        raise ValueError(f"Unsupported loss type {loss_type}")
