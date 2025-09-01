import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import tqdm
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
from utils import TemporarilySeededRandom
from dataset.data_utils import load_json_dataset

def format_dpo_conversation(conversations: List[Dict[str, str]]) -> str:
    """Format DPO conversation messages into a single string.
    
    Args:
        conversations: List of conversation dictionaries with 'from' and 'value' keys
        
    Returns:
        Formatted conversation string as prompt
    """
    formatted_parts = []
    
    for message in conversations:
        role = message['from']
        content = message['value'].strip()
        
        if role == 'system' and content:
            formatted_parts.append(f"system\n{content}\n")
        elif role == 'human':
            formatted_parts.append(f"user\n{content}\n")
        elif role == 'gpt' or role == 'assistant':
            formatted_parts.append(f"assistant\n{content}\n")
    
    # Add final assistant prompt
    prompt = ''.join(formatted_parts) + "assistant\n"
    return prompt


def extract_dpo_data(dpo_example: Dict) -> Tuple[str, str, str]:
    """Extract prompt, chosen response, and rejected response from DPO example."""
    
    # Validate input structure
    for key in ['conversations', 'chosen', 'rejected']:
        if key not in dpo_example:
            raise ValueError(f"Missing '{key}' key in DPO example")
    
    # Format conversation as prompt
    conversations = dpo_example['conversations']
    prompt = format_dpo_conversation(conversations)
    
    # Extract chosen and rejected responses
    chosen_data = dpo_example['chosen']
    rejected_data = dpo_example['rejected']
    
    chosen_response = ' ' + chosen_data['value'].strip()
    rejected_response = ' ' + rejected_data['value'].strip()
    
    # Validate responses are not empty
    if not chosen_response.strip() or not rejected_response.strip():
        raise ValueError(f"Empty response detected - chosen: '{chosen_response}', rejected: '{rejected_response}'")
    
    return prompt, chosen_response, rejected_response


def get_dpo_dataset(file_path: str, split: str = 'train', silent: bool = False) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load DPO dataset and convert to the format expected by trainers.py."""
    if not silent:
        print(f'Loading DPO dataset ({split} split) from {file_path}...')
    
    raw_data = load_json_dataset(file_path, split, silent)
    
    data = {}
    for example in tqdm.tqdm(raw_data, desc='Processing DPO data', disable=silent):
        try:
            prompt, chosen_response, rejected_response = extract_dpo_data(example)
            
            # Create the expected format
            data[prompt] = {
                'responses': [chosen_response, rejected_response],
                'pairs': [(0, 1)],  # chosen is index 0, rejected is index 1
                'sft_target': chosen_response
            }
            
        except Exception as e:
            if not silent:
                print(f"Warning: Skipping DPO example due to error: {e}")
            continue
    
    if not silent:
        print(f'Processed {len(data)} DPO examples')
    
    return data


def tokenize_batch_element(prompt: str, chosen: str, rejected: str, truncation_mode: str, tokenizer, max_length: int, max_prompt_length: int) -> Dict:
    """Tokenize a single batch element."""
    
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)

    assert tokenizer.eos_token_id not in prompt_tokens['input_ids'], f"Prompt contains EOS token: {prompt}"
    assert tokenizer.eos_token_id not in chosen_tokens['input_ids'], f"Chosen response contains EOS token: {chosen}"
    assert tokenizer.eos_token_id not in rejected_tokens['input_ids'], f"Rejected response contains EOS token: {rejected}"

    chosen_tokens['input_ids'].append(tokenizer.eos_token_id)
    chosen_tokens['attention_mask'].append(1)

    rejected_tokens['input_ids'].append(tokenizer.eos_token_id)
    rejected_tokens['attention_mask'].append(1)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    # if combined sequence is too long, truncate the prompt
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        if truncation_mode == 'keep_start':
            prompt_tokens = {k: v[:max_prompt_length] for k, v in prompt_tokens.items()}
        elif truncation_mode == 'keep_end':
            prompt_tokens = {k: v[-max_prompt_length:] for k, v in prompt_tokens.items()}
        else:
            raise ValueError(f'Unknown truncation mode: {truncation_mode}')

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_length:
        chosen_tokens = {k: v[:max_length - max_prompt_length] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_length - max_prompt_length] for k, v in rejected_tokens.items()}

    # Create labels
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch = {}

    batch['prompt'] = prompt
    batch['chosen'] = prompt + chosen
    batch['rejected'] = prompt + rejected
    batch['chosen_response_only'] = chosen
    batch['rejected_response_only'] = rejected

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch[f'{k}_{type_key}'] = tokens

    return batch


def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels'):
                if 'prompt' in k:  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith('_input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                if 'prompt' in k:  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch
    return collate_fn


def get_dpo_batch_iterator(datasets: List[str],
                      tokenizer,
                      split: str = 'train',
                      batch_size: int = 1,
                      shuffle: bool = True,
                      max_length: int = 512,
                      max_prompt_length: int = 128,
                      sft_mode: bool = False,
                      n_epochs: Optional[int] = None,
                      n_examples: Optional[int] = None,
                      seed: int = 0,
                      silent: bool = False,
                      cache_dir: Optional[str] = None) -> Iterator[Dict]:
    """Get an iterator over batches of data."""
    
    assert n_epochs is not None or n_examples is not None, "Must specify either n_epochs or n_examples"
    
    with TemporarilySeededRandom(seed):
        permutation_seeds = iter(np.random.randint(0, 2**32, size=1000000))
        flat_data = []
        for dataset in datasets:
            if dataset.type == "custom_dataset":
                # Load DPO data
                truncation_mode = 'keep_end'  # Default for DPO
                for prompt, data in get_dpo_dataset(dataset.path, split, silent=silent).items():
                    flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))

            elif dataset.type == "huggingface_dataset":
                from dataset.data_utils import get_huggingface_dataset
                truncation_mode = 'keep_end' if dataset.name == 'hh' else 'keep_start'
                for prompt, data in get_huggingface_dataset(dataset.name, split, silent=silent, cache_dir=cache_dir, path = dataset.path).items():
                    flat_data.append((prompt, data['responses'], data['pairs'], data['sft_target'], truncation_mode))
            else:
                raise ValueError("Unknown dataset")

    collate_fn = get_collate_fn(tokenizer)

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
        for prompt, responses, pairs, sft_target, truncation_mode in flat_data:
            if done:
                break

            for p in pairs:
                if done:
                    break
                batch_element = tokenize_batch_element(prompt, responses[p[0]], responses[p[1]], truncation_mode, tokenizer, max_length, max_prompt_length)
                batch.append(batch_element)
                example_idx += 1
                if len(batch) == batch_size:
                    yield collate_fn(batch)
                    if n_examples is not None and example_idx >= n_examples:
                        if not silent:
                            print(f'FINISHED {n_examples} EXAMPLES on {split} split')
                        done = True
                    batch = []
        if done:
            break

        epoch_idx += 1
