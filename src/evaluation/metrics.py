import re
from collections.abc import Callable, Sequence


MetricValues = dict[str, float]
PredictionPair = tuple[str, str]


def evaluate_stage(
    stage_name: str,
    predictions: Sequence[PredictionPair],
) -> MetricValues:
    if not predictions:
        raise ValueError("At least one prediction is required.")

    evaluators: dict[str, Callable[[str, str], MetricValues]] = {
        "stage1_mcq": _mcq_metrics,
        "stage2_eeg_captioning": _caption_metrics,
        "stage3_attention_task_cot": _recording_cot_metrics,
        "stage4_resting_state_cot": _recording_cot_metrics,
        "stage5_diagnostic_cot": _diagnostic_metrics,
    }
    try:
        evaluator = evaluators[stage_name]
    except KeyError as error:
        raise KeyError(f"No metrics are defined for stage '{stage_name}'.") from error

    sample_metrics = [
        evaluator(predicted, expected)
        for predicted, expected in predictions
    ]
    names = sample_metrics[0].keys()
    return {
        name: sum(values[name] for values in sample_metrics) / len(sample_metrics)
        for name in names
    }


def _mcq_metrics(predicted: str, expected: str) -> MetricValues:
    predicted_answer = _extract_mcq(predicted)
    expected_answer = _extract_mcq(expected)
    return {
        "accuracy": float(
            expected_answer is not None and predicted_answer == expected_answer
        ),
    }


def _caption_metrics(predicted: str, expected: str) -> MetricValues:
    return {
        "exact_match": float(_normalize(predicted) == _normalize(expected)),
        "token_f1": _token_f1(predicted, expected),
    }


def _recording_cot_metrics(predicted: str, expected: str) -> MetricValues:
    predicted_conclusion = _section(predicted, "Conclusion", "Confidence")
    expected_conclusion = _section(expected, "Conclusion", "Confidence")
    predicted_confidence = _field(
        predicted,
        "confidence",
        ("low", "moderate", "high"),
    )
    expected_confidence = _field(
        expected,
        "confidence",
        ("low", "moderate", "high"),
    )
    return {
        "exact_match": float(_normalize(predicted) == _normalize(expected)),
        "token_f1": _token_f1(predicted, expected),
        "conclusion_token_f1": _token_f1(
            predicted_conclusion,
            expected_conclusion,
        ),
        "confidence_accuracy": float(
            expected_confidence is not None
            and predicted_confidence == expected_confidence
        ),
    }


def _diagnostic_metrics(predicted: str, expected: str) -> MetricValues:
    fields = {
        "diagnosis": ("NOT_ADHD", "ADHD"),
        "presentation": (
            "predominantly_inattentive",
            "predominantly_hyperactive_impulsive",
            "combined",
            "unspecified",
            "not_applicable",
        ),
        "severity": ("none", "mild", "moderate", "severe", "not_applicable"),
        "confidence": ("low", "moderate", "high"),
    }
    result: MetricValues = {
        "exact_match": float(_normalize(predicted) == _normalize(expected)),
        "token_f1": _token_f1(predicted, expected),
    }
    for field_name, values in fields.items():
        predicted_value = _field(predicted, field_name, values)
        expected_value = _field(expected, field_name, values)
        result[f"{field_name}_accuracy"] = float(
            expected_value is not None and predicted_value == expected_value
        )
    return result


def _extract_mcq(value: str) -> str | None:
    match = re.search(r"(?:^|\bANSWER\s*:\s*)([ABCD])(?:\b|$)", value.upper())
    return match.group(1) if match else None


def _section(value: str, start: str, end: str) -> str:
    match = re.search(
        rf"\b{re.escape(start)}\s*:\s*(.*?)\s+{re.escape(end)}\s*:",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1) if match else ""


def _field(value: str, name: str, choices: tuple[str, ...]) -> str | None:
    alternatives = "|".join(
        re.escape(choice) for choice in sorted(choices, key=len, reverse=True)
    )
    match = re.search(
        rf"\b{re.escape(name)}\s*[: ]\s*({alternatives})\b",
        value,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


def _token_f1(predicted: str, expected: str) -> float:
    predicted_tokens = _tokens(predicted)
    expected_tokens = _tokens(expected)
    if not predicted_tokens or not expected_tokens:
        return float(predicted_tokens == expected_tokens)

    predicted_counts: dict[str, int] = {}
    expected_counts: dict[str, int] = {}
    for token in predicted_tokens:
        predicted_counts[token] = predicted_counts.get(token, 0) + 1
    for token in expected_tokens:
        expected_counts[token] = expected_counts.get(token, 0) + 1
    overlap = sum(
        min(count, expected_counts.get(token, 0))
        for token, count in predicted_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 2 * precision * recall / (precision + recall)


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:_[a-z0-9]+)*", value.lower())


def _normalize(value: str) -> str:
    return " ".join(_tokens(value))
