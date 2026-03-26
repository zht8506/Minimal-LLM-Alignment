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
