import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_codex_executor_maps_full_access_to_writable_mode():
    source = (REPO_ROOT / "examples" / "auto_deploy" / "run_codex_executor.py").read_text(
        encoding="utf-8"
    )

    assert "--full-auto" in source
    assert "workspace-write" in source
    assert "Configured executor permission" in source
    assert "rejects Codex dangerous full-access mode" in source
    assert "conda run -n auto_deploy_flux_eff python -c" in source
    assert "--dangerously-bypass-approvals-and-sandbox" not in source


def test_real_flux_lab_gives_executor_full_access_only():
    lab = (REPO_ROOT / "examples" / "auto_deploy" / "lab_flux_real.yaml").read_text(
        encoding="utf-8"
    )

    assert 'executor_permission: "full_access"' in lab
    assert 'checker_permission: "auto_review"' in lab
    assert 'reviewer_permission: "auto_review"' in lab


def test_visual_quality_codex_call_uses_auto_review_read_only():
    source = (
        REPO_ROOT / "examples" / "auto_deploy" / "visual_quality_from_metrics.py"
    ).read_text(encoding="utf-8")

    assert "--full-auto" in source
    assert "read-only" in source
    assert "--dangerously-bypass-approvals-and-sandbox" not in source


def test_checker_wrapper_rejects_abnormal_evidence(tmp_path, monkeypatch):
    checker = _load_auto_deploy_module("run_checker_from_action.py")
    monkeypatch.setenv("EAL_ITERATION", "3")
    metrics_path = tmp_path / "metrics.json"
    stdout_log = tmp_path / "stdout.txt"
    stderr_log = tmp_path / "stderr.txt"
    converted_path = tmp_path / "converted.json"
    metrics_path.write_text("{}", encoding="utf-8")

    result = checker._scheduler_checker_result(
        raw_result={
            "benchmark_valid": True,
            "recommendation": "accept",
            "speed": {"delta_reuse": {"aggregate_dit_speedup": 11.56}},
            "quality": {"delta_reuse": {"mean_psnr": 27.55}},
            "integrity_checks": {
                "failed": [
                    "Measured speedup is implausibly above theoretical limit."
                ]
            },
            "implementation_issues": [
                "Implementation evidence failed integrity checks."
            ],
        },
        visual_result={},
        metrics_path=metrics_path,
        converted_path=converted_path,
        command="python benchmark.py",
        exit_code=0,
        duration=1.0,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        conversion_error="",
    )

    assert result["benchmark_valid"] is True
    assert result["implementation_valid"] is False
    assert result["recommendation"] == "REJECT"
    assert "same-candidate repair" in result["rationale"]


def test_checker_wrapper_invalidates_wrong_official_config(tmp_path, monkeypatch):
    checker = _load_auto_deploy_module("run_checker_from_action.py")
    monkeypatch.setenv("EAL_ITERATION", "4")
    metrics_path = tmp_path / "metrics.json"
    stdout_log = tmp_path / "stdout.txt"
    stderr_log = tmp_path / "stderr.txt"
    converted_path = tmp_path / "converted.json"
    metrics_path.write_text("{}", encoding="utf-8")

    result = checker._scheduler_checker_result(
        raw_result={
            "benchmark_valid": True,
            "recommendation": "accept",
            "summary": {
                "benchmark_config": {
                    "width": 512,
                    "height": 512,
                    "steps": 4,
                    "baseline_artifact_name": "old_baseline",
                }
            },
        },
        visual_result={},
        action={
            "baseline_policy": {
                "artifact_name": "official_baseline",
                "fixed_config": {
                    "width": 1024,
                    "height": 1024,
                    "steps": 4,
                },
            }
        },
        metrics_path=metrics_path,
        converted_path=converted_path,
        command="python benchmark.py",
        exit_code=0,
        duration=1.0,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        conversion_error="",
    )

    assert result["benchmark_valid"] is False
    assert result["implementation_valid"] is False
    assert result["recommendation"] == "NEEDS_RETEST"
    assert "fixed official model hyperparameters" in result["rationale"]


def _load_auto_deploy_module(filename: str):
    path = REPO_ROOT / "examples" / "auto_deploy" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
