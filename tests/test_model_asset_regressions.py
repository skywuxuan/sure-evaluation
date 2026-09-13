from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


def test_modelscope_download_uses_pinned_revision_and_local_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sure_eval.evaluation.cli import _download_asset

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "modelscope",
        SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs)),
    )
    target = tmp_path / "ExampleModel" / "model.bin"
    _download_asset(
        {
            "provider": "modelscope",
            "id": "example/model",
            "revision": "pinned",
            "layout": "local_dir",
            "target_path": str(target),
        }
    )
    assert calls == [
        {
            "model_id": "example/model",
            "revision": "pinned",
            "local_dir": str(target.parent),
        }
    ]


def test_modelscope_download_preserves_legacy_cache_layout_without_new_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sure_eval.evaluation.cli import _download_asset

    calls = []
    monkeypatch.setitem(
        sys.modules,
        "modelscope",
        SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs)),
    )
    target = tmp_path / "modelscope" / "models" / "iic" / "example" / "model.pt"
    _download_asset(
        {
            "provider": "modelscope",
            "id": "iic/example",
            "target_path": str(target),
        }
    )
    assert calls == [
        {
            "model_id": "iic/example",
            "cache_dir": str(target.parents[1]),
        }
    ]


def test_downloaded_asset_checksum_is_enforced(tmp_path: Path) -> None:
    from sure_eval.evaluation.cli import _verify_downloaded_asset

    target = tmp_path / "model.bin"
    target.write_bytes(b"model")
    expected = hashlib.sha256(b"model").hexdigest()
    _verify_downloaded_asset({"target_path": str(target), "sha256": expected})
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _verify_downloaded_asset({"target_path": str(target), "sha256": "0" * 64})


def test_env_check_preserves_existing_directory_checkpoint_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sure_eval.evaluation import env_check

    checker = env_check.NodeEnvChecker()
    node_path = Path("src/sure_eval/evaluation/nodes/transcription/qwen3_asr_1_7b")
    node_env = checker.load_node_env("transcription/qwen3_asr_1_7b")
    monkeypatch.setenv("QWEN3_ASR_1_7B_CHECKPOINT", "/tmp")

    checkpoint_path, checkpoint_env = checker._checkpoint_path(
        "transcription/qwen3_asr_1_7b",
        node_path,
        node_env,
    )
    assert checkpoint_env == "QWEN3_ASR_1_7B_CHECKPOINT"
    assert checkpoint_path == Path("/tmp")


@pytest.mark.parametrize("declare_sha256", [False, True])
def test_env_check_skips_or_accepts_declared_checkpoint_checksum(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    declare_sha256: bool,
) -> None:
    from sure_eval.evaluation import env_check

    node_dir = tmp_path / "scoring" / "example"
    (node_dir / ".venv" / "bin").mkdir(parents=True)
    (node_dir / ".venv" / "bin" / "python3.11").write_bytes(b"")
    checkpoint = node_dir / "checkpoints" / "model.bin"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model")
    model = {"id": "example/model", "target": "checkpoints/model.bin"}
    if declare_sha256:
        model["sha256"] = hashlib.sha256(b"model").hexdigest()
    (node_dir / "node_env.yaml").write_text(
        yaml.safe_dump(
            {
                "runtime": {"type": "uv", "python": "3.11"},
                "models": [model],
                "verify": {"files": []},
            }
        )
    )
    monkeypatch.setattr(env_check, "load_node_manifest", lambda node_id: ({}, node_dir))

    result = env_check.NodeEnvChecker(nodes_root=tmp_path).check_node("scoring/example")
    assert result.status == "ok"
    if declare_sha256:
        assert result.details["checkpoint_sha256"] == model["sha256"]
    else:
        assert "checkpoint_sha256" not in result.details


def test_env_check_rejects_bad_checkpoint_checksum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sure_eval.evaluation import env_check

    node_dir = tmp_path / "scoring" / "example"
    (node_dir / ".venv" / "bin").mkdir(parents=True)
    (node_dir / ".venv" / "bin" / "python3.11").write_bytes(b"")
    checkpoint = node_dir / "checkpoints" / "model.bin"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"corrupt")
    (node_dir / "node_env.yaml").write_text(
        yaml.safe_dump(
            {
                "runtime": {"type": "uv", "python": "3.11"},
                "models": [
                    {
                        "id": "example/model",
                        "target": "checkpoints/model.bin",
                        "sha256": "0" * 64,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(env_check, "load_node_manifest", lambda node_id: ({}, node_dir))

    result = env_check.NodeEnvChecker(nodes_root=tmp_path).check_node("scoring/example")
    assert result.status == "failed"
    assert "checkpoint checksum failed" in result.message


def test_node_local_import_check_reports_runtime_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from sure_eval.evaluation import env_check

    monkeypatch.setattr(
        env_check.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'missing_runtime_dependency'\n",
        ),
    )
    assert env_check._check_node_local_imports(Path("python"), ["example"]) == (
        "ModuleNotFoundError: No module named 'missing_runtime_dependency'"
    )


def test_node_local_import_check_does_not_inherit_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sure_eval.evaluation import env_check

    observed = {}

    def fake_run(*args, **kwargs):
        observed.update(kwargs["env"])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("PYTHONPATH", "/host/private/modules")
    monkeypatch.setattr(env_check.subprocess, "run", fake_run)
    assert env_check._check_node_local_imports(Path("python"), ["example"]) == ""
    assert "PYTHONPATH" not in observed
    assert observed["PYTHONNOUSERSITE"] == "1"
