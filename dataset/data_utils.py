import json
import datasets
import tqdm
from typing import Dict, List, Tuple, Union
from collections import defaultdict
from bs4 import BeautifulSoup, NavigableString
import os


def load_json_dataset(file_path: str, split: str = 'train', silent: bool = False) -> List[Dict]:
    """Load JSON dataset from file.
    
    Args:
        file_path: Path to the JSON file
        split: Dataset split (train/test/val)
        silent: Whether to silence progress bars
        
    Returns:
        List of conversation data
    """
    if not silent:
        print(f'Loading JSON dataset from {file_path} ({split} split)...')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract the appropriate split if data is a dictionary
    if isinstance(data, dict) and split in data:
        # Try to get the specified split first, then fall back to 'train', 
        # and finally use the entire data if neither exists
        data = data.get(split, data.get('train', data))
    
    if not silent:
        print(f'Loaded {len(data)} examples')
    
    return data


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = '\n\nAssistant:'
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[:search_term_idx + len(search_term)]


def strip_html_tags(html_string):
    """Strip HTML tags from a string, except for <code> tags (which contain real code in the StackExchange answers)."""
    # Create a BeautifulSoup object
    soup = BeautifulSoup(html_string, 'html.parser')

    # Initialize an empty list to store the text
    text = []
    for element in soup.children:
        if isinstance(element, NavigableString):
            continue
        if element.name == 'p':
            text.append(''.join(child.string for child in element.children if isinstance(child, NavigableString)))
        elif element.name == 'pre':
            for code in element.find_all('code'):
                text.append("<code>" + code.get_text() + "</code>")
        elif element.name == 'code':
            text.append("<code>" + element.get_text() + "</code>")

    # Join the text together with newlines in between
    text = "\n\n".join(text)

    return text


def get_se(data_name_path: str, split, silent=False, cache_dir: str = None, path = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the StackExchange dataset from Huggingface, and return a dict of prompts and responses. See get_hh for the format.
    
       We strip the HTML tags from the responses (except for <code> tags), and we add necessary newlines.
    """
    print(f'Loading SE dataset ({split} split) from Huggingface/local_dirs...')

    # if the data_name_path is not a valid path, download the dataset from Huggingface
    load_dataset_path = "HuggingFaceH4/stack-exchange-preferences" if (not path or not os.path.exists(path)) else path
    dataset = datasets.load_dataset(load_dataset_path, cache_dir=cache_dir)['train']

    # shuffle the dataset and select 1% for test
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(int(len(dataset) * 0.01))) if split == 'test' else dataset.select(
        range(int(len(dataset) * 0.01), len(dataset)))

    def strip_html(x):
        x['question'] = strip_html_tags(x['question'])
        for a in x['answers']:
            a['text'] = strip_html_tags(a['text'])
        return x

    dataset = dataset.map(strip_html, num_proc=64)

    data = defaultdict(dict)
    for row in tqdm.tqdm(dataset, desc='Processing SE', disable=silent):
        prompt = '\n\nHuman: ' + row['question'] + '\n\nAssistant:'
        responses = [' ' + a['text'] for a in row['answers']]
        scores = [a['pm_score'] for a in row['answers']]

        pairs = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                pairs.append((i, j) if scores[i] > scores[j] else (j, i))

        data[prompt]['responses'] = responses
        data[prompt]['pairs'] = pairs
        data[prompt]['sft_target'] = max(responses, key=lambda x: scores[responses.index(x)])

    return data


def get_shp(data_name_path: str, split: str, silent: bool = False, cache_dir: str = None, path = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Stanford Human Preferences dataset from Huggingface and convert it to the necessary format. See hh for the format.

        We filter preference pairs to only keep pairs where the score ratio is at least 2.
        For this dataset, the sft_target is the response with the highest score.
    """
    print(f'Loading SHP dataset ({split} split) from Huggingface/local_dirs...')

    # if the data_name_path is not a valid path, download the dataset from Huggingface
    load_dataset_path = "stanfordnlp/SHP" if (not path or not os.path.exists(path)) else path
    dataset = datasets.load_dataset(load_dataset_path, split=split, cache_dir=cache_dir)

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing SHP', disable=silent):
        prompt = '\n\nHuman: ' + row['history'] + '\n\nAssistant:'
        responses = [' ' + row['human_ref_A'], ' ' + row['human_ref_B']]
        scores = [row['score_A'], row['score_B']]
        if prompt in data:
            n_responses = len(data[prompt]['responses'])
        else:
            n_responses = 0
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])
        if score_ratio < 2:
            continue

        # according to https://huggingface.co/datasets/stanfordnlp/SHP 
        data[prompt]['pairs'].append((n_responses, n_responses + 1) if row['labels'] == 1 else (n_responses + 1, n_responses))
        data[prompt]['responses'].extend(responses)
        data[prompt]['scores'].extend(scores)

    for prompt in data:
        data[prompt]['sft_target'] = max(data[prompt]['responses'], key=lambda x: data[prompt]['scores'][data[prompt]['responses'].index(x)])
        del data[prompt]['scores']

    return data


def get_hh(split: str, silent: bool = False, cache_dir: str = None, path = None) -> Dict[str, Dict[str, Union[List[Tuple[int, int]], List[str], str]]]:
    """Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it to the necessary format.
    
       The dataset is converted to a dictionary with the following structure:
       {
           'prompt1': {
               'responses': List[str],
               'pairs': List[Tuple[int, int]],
               'sft_target': str
           },
           'prompt2': {
               ...
           },
       }

       Prompts should be structured as follows:
         \n\nHuman: <prompt>\n\nAssistant:
       Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
       
       For this dataset, the sft_target is just the chosen response.
    """
    print(f'Loading HH dataset ({split} split) from Huggingface/local_dirs...')

    # if the data_name_path is not a valid path, download the dataset from Huggingface
    load_dataset_path = "Anthropic/hh-rlhf" if (not path or not os.path.exists(path)) else path
    dataset = datasets.load_dataset(load_dataset_path, split=split, cache_dir=cache_dir)

    def split_prompt_and_responses(ex):
        prompt = extract_anthropic_prompt(ex['chosen'])
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing HH', disable=silent):
        prompt, chosen, rejected = split_prompt_and_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    return data


def get_huggingface_dataset(name: str, split: str, silent: bool = False, cache_dir: str = None, path = None):
    """Load the given dataset by name. Supported by default are 'shp', 'hh', and 'se'."""
    if name == 'shp':
        data = get_shp(split, silent=silent, cache_dir=cache_dir, path = path)
    elif name == 'hh':
        data = get_hh(split, silent=silent, cache_dir=cache_dir, path = path)
    elif name == 'se':
        data = get_se(split, silent=silent, cache_dir=cache_dir, path = path)
    else:
        raise ValueError(f"Unknown dataset '{name}'")

    assert set(list(data.values())[0].keys()) == {'responses', 'pairs', 'sft_target'}, \
        f"Unexpected keys in dataset: {list(list(data.values())[0].keys())}"

    return data
