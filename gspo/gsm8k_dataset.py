import json
from typing import Any, Dict, List

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class GSM8KJsonDataset(Dataset):
    """
    JSON-based GSM8K dataset loader for PPO/GRPO prompt rollout.
    """

    def __init__(self, json_path: str, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("GSM8K json file must contain a list of samples.")
        self.samples = data

    def __len__(self) -> int:
        return len(self.samples)

    def _build_prompt_text(self, messages: List[Dict[str, str]]) -> str:
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Each sample must contain a non-empty `prompt` list.")
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]

        messages = item.get("prompt", [])
        prompt_text = self._build_prompt_text(messages)

        reward_model = item.get("reward_model", {}) or {}
        ground_truth = str(reward_model.get("ground_truth", "")).strip()

        extra_info = item.get("extra_info", {}) or {}
        question = extra_info.get("question", "")
        answer = extra_info.get("answer", "")

        return {
            "prompt": prompt_text,
            "target": ground_truth,
            "ground_truth": ground_truth,
            "question": question,
            "answer": answer,
            "data_source": item.get("data_source", "gsm8k"),
            "messages": messages,
            "raw_item": item,
        }


def gsm8k_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """
    Collate function compatible with current ppo_train.py:
    - prompts -> List[str]
    - targets -> List[str]
    """
    return {
        "prompts": [b["prompt"] for b in batch],
        "targets": [b["target"] for b in batch],
        "ground_truths": [b["ground_truth"] for b in batch],
        "questions": [b["question"] for b in batch],
        "answers": [b["answer"] for b in batch],
        "data_sources": [b["data_source"] for b in batch],
        "messages": [b["messages"] for b in batch],
        "raw_items": [b["raw_item"] for b in batch],
    }
