import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class JsonSFTDataset(Dataset):
    """Minimal SFT dataset for JSON messages format.
        Example (data/sft_en_demo.json).
    """

    def __init__(self, json_path: str, tokenizer: PreTrainedTokenizerBase, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def _build_ids(self, messages: List[Dict[str, str]]) -> Dict[str, List[int]]:
        if not messages or not isinstance(messages, list):
            raise ValueError("Each sample must contain a non-empty `messages` list.")
        if messages[-1].get("role") != "assistant":
            raise ValueError("The last message must be assistant for SFT.")

        prompt_messages = messages[:-1]
        full_messages = messages

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Keep the tail if too long.
        if len(full_ids) > self.max_length:
            full_ids = full_ids[-self.max_length:]
            prompt_ids = prompt_ids[-self.max_length:]

        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": full_ids, "labels": labels}

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        item = self.samples[idx]
        return self._build_ids(item["messages"])

class SFTDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            ids = f["input_ids"]
            lbs = f["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_len)
            labels.append(lbs + [-100] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
