import re
from typing import Iterable, List


def extract_solution(solution_str: str, method: str = "strict") -> str | None:
    """
    Extract final answer from model output.

    - strict: match GSM8K convention "#### <number>"
    - flexible: fallback to the last valid number in text
    """
    if method not in {"strict", "flexible"}:
        raise ValueError(f"Unsupported method: {method}")

    if method == "strict":
        solutions = re.findall(r"#### (\-?[0-9\.,]+)", solution_str)
        if not solutions:
            return None
        return solutions[-1].replace(",", "").replace("$", "").strip()

    # flexible
    numbers = re.findall(r"(\-?[0-9\.,]+)", solution_str)
    if not numbers:
        return None
    for candidate in reversed(numbers):
        candidate = candidate.strip()
        if candidate not in {"", "."}:
            return candidate.replace(",", "")
    return None


def compute_gsm8k_reward(
    response: str,
    ground_truth: str,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """
    GSM8K rule reward:
    - correct final answer -> score (default 1.0)
    - wrong/format-only answer -> format_score (default 0.0)
    - no extractable answer -> 0.0
    """
    pred = extract_solution(response, method=method)
    if pred is None:
        return 0.0
    if pred == str(ground_truth).strip():
        return float(score)
    return float(format_score)


def compute_gsm8k_reward_batch(
    responses: Iterable[str],
    ground_truths: Iterable[str],
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> List[float]:
    rewards = []
    for response, gt in zip(responses, ground_truths):
        rewards.append(
            compute_gsm8k_reward(
                response=response,
                ground_truth=gt,
                method=method,
                format_score=format_score,
                score=score,
            )
        )
    return rewards
