from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

MATH_DATASETS = {"aime24", "aime25"}
GPQA_DATASETS = {"gpqa", "gpqa_diamond"}


def _normalize_text(value: str) -> str:
    value = value.strip()
    value = value.replace("$", "")
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(".")
    return value


def extract_last_boxed(text: str) -> str | None:
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text)]
    if not starts:
        return None
    start = starts[-1]
    brace = text.find("{", start)
    depth = 0
    for i in range(brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : i]
    return None


def _strip_outer_wrappers(value: str) -> str:
    pairs = {"(": ")", "{": "}", "[": "]"}

    def single_wrapped(text: str, left: str, right: str) -> bool:
        if not (text.startswith(left) and text.endswith(right)):
            return False
        depth = 0
        for idx, char in enumerate(text):
            if char == left:
                depth += 1
            elif char == right:
                depth -= 1
                if depth == 0 and idx != len(text) - 1:
                    return False
        return depth == 0

    changed = True
    while changed and value:
        changed = False
        for left, right in pairs.items():
            if single_wrapped(value, left, right):
                value = value[1:-1].strip()
                changed = True
    return value


def _clean_latex_answer(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    while True:
        if len(text) >= 4 and text.startswith("$$") and text.endswith("$$"):
            text = text[2:-2].strip()
            continue
        if len(text) >= 2 and text.startswith("$") and text.endswith("$"):
            text = text[1:-1].strip()
            continue
        break
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = re.sub(r"\\sqrt\s*([A-Za-z0-9])(?![A-Za-z0-9])", r"\\sqrt{\1}", text)
    return text.strip() or None


def normalize_math_expr(expr: str | None) -> str | None:
    text = _clean_latex_answer(expr)
    if text is None:
        return None
    text = text.replace("$", "").replace(",", "").replace(" ", "")
    text = _strip_outer_wrappers(text)
    text = re.sub(r"\^\{?\\circ\}?$", "", text)
    text = re.sub(r"\^\{?circ\}?$", "", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = _strip_outer_wrappers(text)
    if re.fullmatch(r"[+-]?\d+", text):
        return str(int(text))
    for pattern in [
        re.compile(r"\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}$"),
        re.compile(r"([+-]?\d+)/([+-]?\d+)$"),
    ]:
        match = pattern.fullmatch(text)
        if not match:
            continue
        den = int(match.group(2))
        if den == 0:
            return text
        frac = Fraction(int(match.group(1)), den)
        return str(frac.numerator) if frac.denominator == 1 else f"{frac.numerator}/{frac.denominator}"
    return text


def extract_gsm8k_number(text: str) -> str | None:
    marker = re.findall(r"####\s*([^\n]+)", text)
    if marker:
        return marker[-1].strip()
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


def _numeric_equal(a: str, b: str) -> bool:
    try:
        return Decimal(a.replace(",", "")) == Decimal(b.replace(",", ""))
    except (InvalidOperation, ValueError):
        return False


def _math_verify_match(prediction: str, gold: str) -> bool | None:
    try:
        from math_verify import parse, verify as math_verify
    except Exception:
        return None
    try:
        return bool(math_verify(parse(gold), parse(prediction)))
    except Exception:
        return None


def _sympy_match(prediction: str, gold: str) -> bool | None:
    try:
        import sympy as sp
    except Exception:
        return None
    try:
        pred_expr = sp.sympify(prediction.replace("^", "**"))
        gold_expr = sp.sympify(gold.replace("^", "**"))
        return bool(sp.simplify(pred_expr - gold_expr) == 0)
    except Exception:
        return None


def extract_choice_letter(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    boxed = extract_last_boxed(text)
    for candidate in [boxed, text]:
        if not candidate:
            continue
        match = re.search(r"\b([ABCD])\b", candidate.upper())
        if match:
            return match.group(1)
    return None


def _extract_math_prediction(output: str) -> str | None:
    boxed = extract_last_boxed(output)
    if boxed is not None:
        return boxed
    labeled = re.findall(
        r"(?is)\b(?:final\s+answer|answer|the\s+answer|result)\b\s*(?:is|=|:)?\s*([^\n]+)",
        output,
    )
    if labeled:
        return labeled[-1].strip()
    return extract_gsm8k_number(output)


def math_match(output: str | None, gold: str | None) -> tuple[bool, str | None]:
    if output is None or gold is None:
        return False, None
    prediction = _extract_math_prediction(output)
    pred_norm = normalize_math_expr(prediction)
    gold_norm = normalize_math_expr(gold)
    if pred_norm is None or gold_norm is None:
        return False, prediction
    verified = _math_verify_match(pred_norm, gold_norm)
    if verified is not None:
        return verified, prediction
    sympy_verified = _sympy_match(pred_norm, gold_norm)
    if sympy_verified is not None:
        return sympy_verified, prediction
    return pred_norm == gold_norm or _numeric_equal(pred_norm, gold_norm), prediction


def gpqa_match(output: str | None, gold: str | None) -> tuple[bool, str | None]:
    prediction = extract_choice_letter(output)
    gold_letter = extract_choice_letter(gold) or (str(gold).strip().upper()[:1] if gold else None)
    return prediction is not None and prediction == gold_letter, prediction


def _auto_verifier(verifier: str, dataset: str | None) -> str:
    if verifier not in {"auto", "benchmark"}:
        return verifier
    if dataset in MATH_DATASETS:
        return "math"
    if dataset in GPQA_DATASETS:
        return "gpqa"
    return "boxed"


def verify(output: str, gold: str | None, verifier: str, dataset: str | None = None) -> dict:
    if gold is None or verifier == "none":
        return {"correct": None, "prediction": None, "gold": gold}

    verifier = _auto_verifier(verifier, dataset)
    if verifier == "exact":
        prediction = output.strip()
    elif verifier == "boxed":
        prediction = extract_last_boxed(output)
        if prediction is None:
            prediction = extract_gsm8k_number(output)
    elif verifier == "gsm8k_numeric":
        prediction = extract_gsm8k_number(output)
    elif verifier in {"math", "aime"}:
        correct, prediction = math_match(output, gold)
        return {"correct": bool(correct), "prediction": prediction, "gold": gold}
    elif verifier == "gpqa":
        correct, prediction = gpqa_match(output, gold)
        return {"correct": bool(correct), "prediction": prediction, "gold": gold}
    else:
        raise ValueError(f"Unknown verifier: {verifier}")

    if prediction is None:
        correct = False
    else:
        norm_pred = _normalize_text(prediction)
        norm_gold = _normalize_text(gold)
        correct = norm_pred == norm_gold or _numeric_equal(norm_pred, norm_gold)
    return {"correct": bool(correct), "prediction": prediction, "gold": gold}
