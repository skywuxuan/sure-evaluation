from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_key_labels(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{key}\t{label}\n" for key, label in rows), encoding="utf-8")


def test_lid_label_normalization_canonicalizes_common_variants() -> None:
    from sure_eval.evaluation.nodes.normalization.lid_label import normalize_lid_label

    assert normalize_lid_label(" zh Mandarin ") == "zh-mandarin"
    assert normalize_lid_label("ZH_yue") == "zh-yue"
    assert normalize_lid_label("<unk>") == ""
    assert normalize_lid_label("en") == "en"


def test_lid_pipeline_scores_any_system_label_files(tmp_path: Path) -> None:
    from sure_eval.evaluation.tasks.lid import evaluate_lid_files

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_labels(
        ref_file,
        [("zh", "zh-mandarin"), ("en", "en"), ("fr", "fr"), ("custom", "x-demo")],
    )
    _write_key_labels(
        hyp_file,
        [("zh", "ZH mandarin"), ("en", "en"), ("fr", "de"), ("custom", "X_demo")],
    )

    report = evaluate_lid_files(str(ref_file), str(hyp_file))

    assert report.score == pytest.approx(3 / 4)
    assert report.pipeline_id == "lid.any.accuracy.lid_label_canonical_v1.classify_v1"
    assert report.computation_node_ids == (
        "normalization/lid_label",
        "scoring/classify",
    )
    assert [node.stage for node in report.pipeline_trace] == ["normalization", "scoring"]
    assert report.input_files.as_dict() == {"ref": str(ref_file), "hyp": str(hyp_file)}
    assert report.details["rows"][0]["predicted_language"] == "zh-mandarin"


def test_lid_has_one_evaluation_only_route(tmp_path: Path) -> None:
    from sure_eval.evaluation.scripts.lid import describe_pipeline, run

    pipeline_id = "lid.any.accuracy.lid_label_canonical_v1.classify_v1"
    description = describe_pipeline()
    assert description.pipeline_id == pipeline_id
    assert description.required_roles == ("hyp", "ref")
    assert description.node_ids == ("normalization/lid_label", "scoring/classify")

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    _write_key_labels(ref_file, [("utt1", "en")])
    _write_key_labels(hyp_file, [("utt1", "en")])
    report = run(
        str(ref_file),
        str(hyp_file),
        output_dir=str(tmp_path / "out"),
        pipeline_id=pipeline_id,
    )
    assert report.score == 1.0
    assert json.loads((tmp_path / "out" / "report.json").read_text())["pipeline_id"] == pipeline_id


def test_lid_cli_describe_and_run_label_files(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from sure_eval.cli import app

    ref_file = tmp_path / "ref.txt"
    hyp_file = tmp_path / "hyp.txt"
    pipeline_path = tmp_path / "pipeline.json"
    _write_key_labels(ref_file, [("utt1", "zh-mandarin"), ("utt2", "en")])
    _write_key_labels(hyp_file, [("utt1", "zh mandarin"), ("utt2", "en")])
    runner = CliRunner()

    routes_result = runner.invoke(app, ["metric", "routes", "lid", "--json"])
    assert routes_result.exit_code == 0, routes_result.stdout
    routes = json.loads(routes_result.stdout)
    assert routes["count"] == 1
    assert routes["routes"][0]["required_roles"] == ["hyp", "ref"]

    describe_result = runner.invoke(
        app,
        ["metric", "describe", "lid", "--output", str(pipeline_path), "--json"],
    )
    assert describe_result.exit_code == 0, describe_result.stdout

    run_result = runner.invoke(
        app,
        [
            "metric",
            "run",
            "--pipeline",
            str(pipeline_path),
            "--ref-file",
            str(ref_file),
            "--hyp-file",
            str(hyp_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--json",
        ],
    )
    assert run_result.exit_code == 0, run_result.stdout
    assert json.loads(run_result.stdout)["score"] == 1.0


def test_lid_agent_plan_is_model_independent() -> None:
    from sure_eval.evaluation.agent_plan import build_agent_plan

    payload = build_agent_plan("lid", metric="accuracy", include_root_env=False)
    route = payload["selected_routes"][0]
    assert route["pipeline_id"] == "lid.any.accuracy.lid_label_canonical_v1.classify_v1"
    assert route["required_roles"] == ["hyp", "ref"]
    assert route["computation_node_ids"] == [
        "normalization/lid_label",
        "scoring/classify",
    ]
    assert route["setup_required"] is False
    assert route["can_run_now"] is True
