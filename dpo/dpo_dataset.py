import json
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class JsonDPODataset(Dataset):
    """DPO preference data: shared prompt + chosen vs rejected completions.
        Example (data/dpo_en_demo.json).
    """

    def __init__(self, json_path: str, tokenizer: PreTrainedTokenizerBase, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(json_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def _build_ids(self, messages: List[Dict[str, str]]) -> Dict[str, List[int]]:

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
            full_ids = full_ids[-self.max_length :]
            prompt_ids = prompt_ids[-self.max_length :]

        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {"input_ids": full_ids, "labels": labels}

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        item = self.samples[idx]

        prompt_messages = [{"role": m["from"], "content": m["content"]} for m in item["conversations"]]

        chosen_msg = {"role": item["chosen"]["from"], "content": item["chosen"]["content"]}
        rejected_msg = {"role": item["rejected"]["from"], "content": item["rejected"]["content"]}

        chosen_branch = self._build_ids(prompt_messages + [chosen_msg])
        rejected_branch = self._build_ids(prompt_messages + [rejected_msg])

        return {
            "chosen_input_ids": chosen_branch["input_ids"],
            "chosen_labels": chosen_branch["labels"],
            "rejected_input_ids": rejected_branch["input_ids"],
            "rejected_labels": rejected_branch["labels"],
        }

class DPODataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")

        max_len = 0
        for f in features:
            max_len = max(max_len, len(f["chosen_input_ids"]), len(f["rejected_input_ids"]))

        def pad_branch(prefix: str) -> tuple:
            input_ids, labels, attention_mask = [], [], []
            for f in features:
                ids = f[f"{prefix}_input_ids"]
                lbs = f[f"{prefix}_labels"]
                pad_len = max_len - len(ids)
                input_ids.append(ids + [pad_id] * pad_len)
                labels.append(lbs + [-100] * pad_len)
                attention_mask.append([1] * len(ids) + [0] * pad_len)
            return (
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long),
                torch.tensor(attention_mask, dtype=torch.long),
            )

        chosen_input_ids, chosen_labels, chosen_attention_mask = pad_branch("chosen")
        rejected_input_ids, rejected_labels, rejected_attention_mask = pad_branch("rejected")

        return {
            "chosen_input_ids": chosen_input_ids,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": rejected_attention_mask,
        }
