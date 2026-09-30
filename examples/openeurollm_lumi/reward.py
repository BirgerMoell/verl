"""Standalone exact reward for the integer/rational math qualification."""

from __future__ import annotations

from fractions import Fraction
import re


def _last_boxed(text: str) -> str | None:
    start = text.rfind(r"\boxed{")
    if start < 0:
        return None
    begin = start + len(r"\boxed{")
    depth = 1
    for index in range(begin, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:index]
    return None


def _number(value: str) -> Fraction | None:
    clean = value.strip().replace(" ", "").replace(r"\left", "").replace(r"\right", "")
    latex_fraction = re.fullmatch(r"\\(?:dfrac|tfrac|frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", clean)
    if latex_fraction:
        numerator, denominator = latex_fraction.groups()
        try:
            return Fraction(int(numerator), int(denominator))
        except ZeroDivisionError:
            return None
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:/[+-]?\d+)?", clean):
        return None
    try:
        return Fraction(clean)
    except (ValueError, ZeroDivisionError):
        return None


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    """Return 1 only for a complete final boxed number equal to the ground truth."""
    answer = _last_boxed(solution_str)
    if answer is None:
        return 0.0
    predicted = _number(answer)
    expected = _number(str(ground_truth))
    return float(predicted is not None and expected is not None and predicted == expected)
