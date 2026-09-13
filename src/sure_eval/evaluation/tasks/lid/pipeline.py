"""LID task pipeline for classification accuracy over language labels."""

from __future__ import annotations

from sure_eval.evaluation.core.types import EvaluationFiles, EvaluationReport, MetricInputContract
from sure_eval.evaluation.nodes.normalization.lid_label import normalize_lid_rows
from sure_eval.evaluation.nodes.scoring.classify import (
    load_label_spec,
    read_classification_rows,
    score_classification_rows_node,
)
from sure_eval.evaluation.pipeline_identity import (
    build_atomic_pipeline_id,
    component_trace_ids,
    node_component,
)

_LID_LABEL_CONTRACT = MetricInputContract(
    metric_id="task/lid_key_labels",
    required_roles=("hyp", "ref"),
    row_format="key_label",
    alignment_key="key",
    aggregation="utterance_accuracy",
    purpose="spoken_language_identification_accuracy",
)


def evaluate_lid_files(ref_file: str, hyp_file: str) -> EvaluationReport:
    """Score labels produced by any LID system against reference labels."""

    input_files = EvaluationFiles.from_ref_hyp(ref_file, hyp_file)
    _LID_LABEL_CONTRACT.validate(input_files)
    references = read_classification_rows(ref_file)
    hypotheses = read_classification_rows(hyp_file)
    normalized_refs, normalized_hyps, normalization_result = normalize_lid_rows(
        references,
        hypotheses,
    )
    scoring_result = score_classification_rows_node(
        normalized_refs,
        normalized_hyps,
        label_spec=_lid_label_spec(normalized_refs, normalized_hyps),
    )
    result = scoring_result.details["result"]
    components = (
        node_component("normalization/lid_label", profile="canonical"),
        node_component("scoring/classify"),
    )
    return EvaluationReport(
        task="LID",
        language="n/a",
        metric="accuracy",
        score=float(result["score"]),
        pipeline_id=build_atomic_pipeline_id("lid", "any", "accuracy", components),
        pipeline_trace=(normalization_result, scoring_result),
        input_contract=_LID_LABEL_CONTRACT,
        input_files=input_files,
        computation_node_ids=component_trace_ids(components),
        details={
            "results": {"accuracy": result},
            "rows": [_label_report_row(row) for row in result["per_sample"]],
            "input_summary": {
                "num_references": len(references),
                "num_predictions": len(hypotheses),
                "reference_languages": sorted({label for _, label in normalized_refs if label}),
            },
            "input_contract": _LID_LABEL_CONTRACT.as_dict(),
            "input_files": input_files.as_dict(),
        },
    )


def _lid_label_spec(
    references: list[tuple[str, str]],
    predictions: list[tuple[str, str]],
):
    labels = tuple(sorted({label for _, label in (*references, *predictions) if label}))
    return load_label_spec(
        {
            "id": "lid_canonical_language_codes",
            "task": "LID",
            "labels": [{"id": code} for code in labels],
            "unknown_policy": "invalid",
            "normalization": {"case_sensitive": True, "strip_punctuation": False},
        }
    )


def _label_report_row(row: dict) -> dict:
    return {
        "sample_id": row["key"],
        "reference_language": row["reference"],
        "predicted_language": row["prediction"],
        "reference_valid": row["reference_valid"],
        "prediction_valid": row["prediction_valid"],
        "correct": row["correct"],
    }
