"""Input parsing and contract checks owned by VAD timebase normalization."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DETECTION_METRICS = ("f1", "p_fa", "p_miss", "dcf_nist")
AUC_METRICS = ("auc_roc",)
ALL_PRIMARY_METRICS = DETECTION_METRICS + AUC_METRICS

REFERENCE_FIELDS = {"key", "duration", "speech_segments"}
PREDICTION_FIELDS = {"key", "speech_segments", "frame_scores"}
SEGMENT_FIELDS = {"start", "end"}
FRAME_SCORE_FIELDS = {"start", "end", "score"}
UNSUPPORTED_SCORE_ALIASES = {"scores", "probs", "speech_probabilities"}
REQUIRED_FIELDS_BY_METRIC = {
    **{metric: ("speech_segments",) for metric in DETECTION_METRICS},
    **{metric: ("frame_scores",) for metric in AUC_METRICS},
}


@dataclass(frozen=True)
class Segment:
    """Half-open speech interval on the reference seconds timebase."""

    start: float
    end: float

    def as_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class FrameScore:
    """Score interval used by frame-score VAD metrics."""

    start: float
    end: float
    score: float

    def as_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end, "score": self.score}


@dataclass(frozen=True)
class VADInputRow:
    """One aligned VAD reference/prediction pair."""

    key: str
    duration: float
    reference_segments: list[Segment]
    prediction_segments: list[Segment] | None
    frame_scores: list[FrameScore] | None
    available_metrics: set[str]
    skipped_metrics: dict[str, str]


@dataclass(frozen=True)
class VADInputBundle:
    """Parsed VAD rows plus input-level summary metadata."""

    rows: list[VADInputRow]
    input_summary: dict[str, Any]


def load_vad_inputs(
    reference_jsonl: str | Path,
    sample_output: str | Path,
    *,
    required_prediction_fields: tuple[str, ...] = (),
) -> VADInputBundle:
    """Parse aligned VAD JSONL files and enforce the documented contract."""

    reference_path = Path(reference_jsonl)
    prediction_path = Path(sample_output)
    reference_rows = _read_jsonl_objects(reference_path, role="reference_jsonl")
    prediction_rows = _read_jsonl_objects(prediction_path, role="sample_output")

    reference_by_key = _rows_by_key(reference_rows, role="reference_jsonl")
    prediction_by_key = _rows_by_key(prediction_rows, role="sample_output")
    _check_aligned_keys(reference_by_key, prediction_by_key)

    rows: list[VADInputRow] = []
    for reference in reference_rows:
        reference_key = _required_str(reference, "key", role="reference_jsonl")
        rows.append(
            _parse_pair(
                reference,
                prediction_by_key[reference_key],
                required_prediction_fields=required_prediction_fields,
            )
        )

    input_summary = {
        "num_rows": len(rows),
        "reference_jsonl": str(reference_path),
        "sample_output": str(prediction_path),
        "num_rows_with_prediction_segments": sum(
            row.prediction_segments is not None for row in rows
        ),
        "num_rows_with_frame_scores": sum(row.frame_scores is not None for row in rows),
        "num_detection_available": sum(
            all(metric in row.available_metrics for metric in DETECTION_METRICS)
            for row in rows
        ),
        "num_auc_available": sum("auc_roc" in row.available_metrics for row in rows),
        "skipped_metrics": _summarize_skipped_metrics(rows),
    }
    return VADInputBundle(rows=rows, input_summary=input_summary)


def _parse_pair(
    reference: dict[str, Any],
    prediction: dict[str, Any],
    *,
    required_prediction_fields: tuple[str, ...],
) -> VADInputRow:
    _reject_unknown_fields(reference, REFERENCE_FIELDS, role="reference_jsonl")
    _reject_score_aliases(prediction)
    _reject_unknown_fields(prediction, PREDICTION_FIELDS, role="sample_output")

    key = _required_str(reference, "key", role="reference_jsonl")
    prediction_key = _required_str(prediction, "key", role="sample_output")
    if prediction_key != key:
        raise ValueError(f"VAD key mismatch: reference {key!r}, prediction {prediction_key!r}")

    duration = _required_finite_float(reference, "duration", role=f"reference_jsonl[{key}]")
    if duration <= 0.0:
        raise ValueError(f"reference_jsonl[{key}].duration must be positive")

    if "speech_segments" not in reference:
        raise ValueError(f"reference_jsonl[{key}] is missing required field: speech_segments")
    reference_segments = _parse_segments(
        reference["speech_segments"],
        role=f"reference_jsonl[{key}].speech_segments",
        duration=duration,
    )

    for field in required_prediction_fields:
        if field not in prediction:
            raise ValueError(f"sample_output[{key}] is missing required field: {field}")

    prediction_segments = None
    if "speech_segments" in prediction:
        prediction_segments = _parse_segments(
            prediction["speech_segments"],
            role=f"sample_output[{key}].speech_segments",
            duration=duration,
        )

    frame_scores = None
    if "frame_scores" in prediction:
        frame_scores = _parse_frame_scores(
            prediction["frame_scores"],
            role=f"sample_output[{key}].frame_scores",
            duration=duration,
        )
        _reject_overlapping_frame_scores(
            frame_scores,
            role=f"sample_output[{key}].frame_scores",
        )

    available_metrics: set[str] = set()
    skipped_metrics: dict[str, str] = {}
    if prediction_segments is None:
        for metric in DETECTION_METRICS:
            skipped_metrics[metric] = "missing prediction field: speech_segments"
    else:
        available_metrics.update(DETECTION_METRICS)

    if frame_scores is None:
        skipped_metrics["auc_roc"] = "missing prediction field: frame_scores"
    else:
        available_metrics.add("auc_roc")

    return VADInputRow(
        key=key,
        duration=duration,
        reference_segments=reference_segments,
        prediction_segments=prediction_segments,
        frame_scores=frame_scores,
        available_metrics=available_metrics,
        skipped_metrics=skipped_metrics,
    )


def _read_jsonl_objects(path: Path, *, role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{role}:{line_number} is not valid JSON") from exc
            if not isinstance(payload, dict):
                raise TypeError(f"{role}:{line_number} must be a JSON object")
            rows.append(payload)
    if not rows:
        raise ValueError(f"{role} must contain at least one JSONL row")
    return rows


def _rows_by_key(rows: list[dict[str, Any]], *, role: str) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows, start=1):
        key = _required_str(row, "key", role=f"{role}:{index}")
        if key in by_key:
            raise ValueError(f"{role} contains duplicate key: {key!r}")
        by_key[key] = row
    return by_key


def _check_aligned_keys(
    reference_by_key: dict[str, dict[str, Any]],
    prediction_by_key: dict[str, dict[str, Any]],
) -> None:
    reference_keys = set(reference_by_key)
    prediction_keys = set(prediction_by_key)
    missing = sorted(reference_keys - prediction_keys)
    extra = sorted(prediction_keys - reference_keys)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing prediction key(s): {', '.join(missing)}")
        if extra:
            details.append(f"unexpected prediction key(s): {', '.join(extra)}")
        raise ValueError("VAD key alignment failed: " + "; ".join(details))


def _reject_unknown_fields(row: dict[str, Any], allowed: set[str], *, role: str) -> None:
    unknown = sorted(set(row) - allowed)
    if unknown:
        raise ValueError(f"{role} contains unsupported field(s): {', '.join(unknown)}")


def _reject_score_aliases(row: dict[str, Any]) -> None:
    aliases = sorted(set(row) & UNSUPPORTED_SCORE_ALIASES)
    if aliases:
        raise ValueError(
            "VAD predictions must use frame_scores; unsupported score alias(es): "
            + ", ".join(aliases)
        )


def _parse_segments(value: Any, *, role: str, duration: float) -> list[Segment]:
    if not isinstance(value, list):
        raise TypeError(f"{role} must be a list")
    segments: list[Segment] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"{role}[{index}] must be an object")
        _reject_unknown_fields(item, SEGMENT_FIELDS, role=f"{role}[{index}]")
        segment = Segment(
            start=_required_finite_float(item, "start", role=f"{role}[{index}]"),
            end=_required_finite_float(item, "end", role=f"{role}[{index}]"),
        )
        _check_interval(segment.start, segment.end, role=f"{role}[{index}]", duration=duration)
        segments.append(segment)
    _reject_overlapping_segments(segments, role=role)
    return segments


def _parse_frame_scores(value: Any, *, role: str, duration: float) -> list[FrameScore]:
    if not isinstance(value, list):
        raise TypeError(f"{role} must be a list")
    frame_scores: list[FrameScore] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"{role}[{index}] must be an object")
        _reject_unknown_fields(item, FRAME_SCORE_FIELDS, role=f"{role}[{index}]")
        frame_score = FrameScore(
            start=_required_finite_float(item, "start", role=f"{role}[{index}]"),
            end=_required_finite_float(item, "end", role=f"{role}[{index}]"),
            score=_required_finite_float(item, "score", role=f"{role}[{index}]"),
        )
        _check_interval(
            frame_score.start,
            frame_score.end,
            role=f"{role}[{index}]",
            duration=duration,
        )
        frame_scores.append(frame_score)
    return frame_scores


def _check_interval(start: float, end: float, *, role: str, duration: float) -> None:
    if start < 0.0 or end > duration:
        raise ValueError(f"{role} must be within [0, duration]")
    if end <= start:
        raise ValueError(f"{role}.end must be greater than start")


def _reject_overlapping_segments(segments: list[Segment], *, role: str) -> None:
    ordered = sorted(segments, key=lambda item: (item.start, item.end))
    previous_end: float | None = None
    for segment in ordered:
        if previous_end is not None and segment.start < previous_end - 1e-12:
            raise ValueError(f"{role} contains overlapping intervals")
        previous_end = max(previous_end or segment.end, segment.end)


def _reject_overlapping_frame_scores(frame_scores: list[FrameScore], *, role: str) -> None:
    ordered = sorted(frame_scores, key=lambda item: (item.start, item.end))
    previous_end: float | None = None
    for frame_score in ordered:
        if previous_end is not None and frame_score.start < previous_end - 1e-12:
            raise ValueError(f"{role} contains overlapping score intervals")
        previous_end = max(previous_end or frame_score.end, frame_score.end)


def _required_str(row: dict[str, Any], field: str, *, role: str) -> str:
    if field not in row:
        raise ValueError(f"{role} is missing required field: {field}")
    value = row[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{role}.{field} must be a non-empty string")
    return value


def _required_finite_float(row: dict[str, Any], field: str, *, role: str) -> float:
    if field not in row:
        raise ValueError(f"{role} is missing required field: {field}")
    value = row[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{role}.{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{role}.{field} must be a finite number")
    return result


def _summarize_skipped_metrics(rows: list[VADInputRow]) -> dict[str, int]:
    counts = {metric: 0 for metric in ALL_PRIMARY_METRICS}
    for row in rows:
        for metric in row.skipped_metrics:
            counts[metric] += 1
    return counts
