import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import tqdm
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union
from utils import TemporarilySeededRandom
from dataset.data_utils import load_json_dataset


def extract_prompt_and_response(messages: List[Dict[str, str]]) -> tuple[str, str]:
    """Extract prompt and response from conversation messages.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Tuple of (prompt, response)
    """
    # Find the last assistant message as the target response
    assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']
    if not assistant_messages:
        raise ValueError("No assistant message found in conversation")
    
    target_response = assistant_messages[-1]['content'].strip()
    
    # Create prompt by excluding the last assistant message
    prompt_messages = []
    for msg in messages:
        if msg['role'] == 'assistant' and msg['content'].strip() == target_response:
            break
        prompt_messages.append(msg)
    
    # Format prompt
    prompt_parts = []
    for message in prompt_messages:
        role = message['role']
        content = message['content'].strip()
        
        if role == 'system' and content:
            prompt_parts.append(f"system\n{content}\n")
        elif role == 'user':
            prompt_parts.append(f"user\n{content}\n")
        elif role == 'assistant':
            prompt_parts.append(f"assistant\n{content}\n")
    
    prompt = ''.join(prompt_parts) + "assistant\n"
    
    return prompt, target_response


def tokenize_sft_batch_element(prompt: str, response: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single SFT batch element directly from prompt and response.

    Returns:
        Dictionary containing tokenized data
    """

    # Tokenize prompt and response separately
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    response_tokens = tokenizer(response, add_special_tokens=False)
    
    # Add EOS token to response
    response_tokens['input_ids'].append(tokenizer.eos_token_id)
    response_tokens['attention_mask'].append(1)
    
    # Handle truncation if sequence is too long
    if len(prompt_tokens['input_ids']) + len(response_tokens['input_ids']) > max_length:
        # First truncate prompt if it's too long
        if len(prompt_tokens['input_ids']) > max_prompt_length:
            if truncation_mode == 'keep_start':
                prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
            elif truncation_mode == 'keep_end':
                prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')
        
        # Then truncate response if still too long
        remaining_length = max_length - len(prompt_tokens['input_ids'])
        if len(response_tokens['input_ids']) > remaining_length:
            response_tokens = {k: v[:remaining_length] for k, v in response_tokens.items()}
    
    # Combine prompt and response
    full_tokens = {
        'input_ids': prompt_tokens['input_ids'] + response_tokens['input_ids'],
        'attention_mask': prompt_tokens['attention_mask'] + response_tokens['attention_mask']
    }
    
    # Create labels (mask prompt tokens with -100)
    labels = full_tokens['input_ids'][:]
    labels[:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    
    batch_element = {
        'input_ids': full_tokens['input_ids'],
        'attention_mask': full_tokens['attention_mask'],
        'labels': labels,
        'response_with_prompt': prompt + response,
        'prompt': prompt,
        'response': response
    }
    
    return batch_element


def get_sft_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for SFT data.
    
    Args:
        tokenizer: Tokenizer to use for padding
        
    Returns:
        Collate function
    """
    def collate_fn(batch):
        padded_batch = {}
        
        for k in batch[0].keys():
            if k.endswith('input_ids') or k.endswith('attention_mask') or k.endswith('labels'):
                # Convert to tensors
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                
                # Determine padding value
                if k.endswith('input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('attention_mask'):
                    padding_value = 0
                elif k.endswith('labels'):
                    padding_value = -100
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")
                
                # Pad sequences
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
            else:
                # Keep string data as lists
                padded_batch[k] = [ex[k] for ex in batch]
        
        return padded_batch
    
    return collate_fn


def get_sft_batch_iterator(datasets,
                      tokenizer,
                      split: str = 'train',
                      batch_size: int = 1,
                      shuffle: bool = True,
                      max_length: int = 512,
                      max_prompt_length: int = 128,
                      sft_mode: bool = True,  # Always True for SFT
                      n_epochs: Optional[int] = None,
                      n_examples: Optional[int] = None,
                      seed: int = 0,
                      silent: bool = False,
                      cache_dir: Optional[str] = None) -> Iterator[Dict]:
    """Compatibility wrapper for trainers.py integration - using SFT mode."""

    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"

    # Use the preference batch iterator in SFT mode
    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for dataset in datasets:
            if dataset.type == "custom_dataset":
                # Load SFT data
                raw_data = load_json_dataset(dataset.path, split, silent)
                truncation_mode = 'keep_end'
                for example in tqdm.tqdm(raw_data, desc='Converting SFT data', disable=silent):
                    messages = example['messages']
                    prompt, response = extract_prompt_and_response(messages)
                    flat_data.append((prompt, response, truncation_mode))

            elif dataset.type == "huggingface_dataset":
                from dataset.data_utils import get_huggingface_dataset
                truncation_mode = 'keep_end' if dataset.name == 'hh' else 'keep_start'
                for prompt, data in get_huggingface_dataset(dataset.name, split, silent=silent, cache_dir=cache_dir, path = dataset.path).items():
                    flat_data.append((prompt, data['sft_target'], truncation_mode))
            else:
                raise ValueError("Unknown dataset")

    collate_fn = get_sft_collate_fn(tokenizer)

    epoch_idx = 0
    example_idx = 0
    done = False
    while True:
        if n_epochs is not None and epoch_idx >= n_epochs:
            if not silent:
                print(f'Finished generating {n_epochs} epochs on {split} split')
            break
        if shuffle:
            with TemporarilySeededRandom(next(permutation_seeds)):
                random.shuffle(flat_data)

        batch = []
        # for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
        for prompt, response, truncation_mode in flat_data:
            if done:
                break
            batch_element = tokenize_sft_batch_element(prompt, response, truncation_mode, tokenizer, max_length, max_prompt_length)
            batch.append(batch_element)
            example_idx += 1
            if len(batch) == batch_size:
                yield collate_fn(batch)
                if n_examples is not None and example_idx >= n_examples:
                    if not silent:
                        print(f'Finished generating {n_examples} examples on {split} split')
                    done = True

                batch = []
        if done:
            break

        epoch_idx += 1
