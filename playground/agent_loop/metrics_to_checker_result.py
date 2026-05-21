from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "auto_deploy.checker_result.v1"
MASKED_QUANTIZATION_SKILLS = {
    "modelopt_tensorrt_fp8_fp4",
    "standalone_svdquant_linear",
    "nunchaku_extracted_linear",
    "nunchaku_svdquant_backend_spec",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _valid_psnr(value: Any) -> bool:
    return isinstance(value, (int, float)) and (math.isfinite(float(value)) or float(value) == float("inf"))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _detect_benchmark_type(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    if rows and "official_kernel_calls" in rows[0]:
        return "sparse_attention"
    if rows and "variant_skipped_steps" in rows[0]:
        return "cache"
    if summary.get("masked_skills") is not None or summary.get("unsupported") is not None:
        return "quantization"
    if rows and any(k.startswith("selective_") for k in rows[0]):
        return "quantization"
    return "generic"


def _path_exists(path_value: Any, metrics_path: Path, repo_root: Path | None) -> bool:
    if not isinstance(path_value, str) or not path_value:
        return False
    path = Path(path_value)
    if path.is_absolute():
        return path.exists()
    candidates = [metrics_path.parent / path]
    if repo_root is not None:
        candidates.append(repo_root / path)
    return any(candidate.exists() for candidate in candidates)


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("variant", "unknown")), []).append(row)
    return grouped


def _validate_rows(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    metrics_path: Path,
    repo_root: Path | None,
    check_images: bool,
) -> tuple[list[str], list[str]]:
    passed: list[str] = []
    failed: list[str] = []
    if not isinstance(summary, dict):
        failed.append("metrics.summary is missing or not an object")
        return passed, failed
    if not rows:
        failed.append("metrics.rows is empty")
        return passed, failed
    passed.append("metrics.rows is non-empty")

    expected_num_rows = summary.get("num_rows")
    if expected_num_rows is None or int(expected_num_rows) == len(rows):
        passed.append("summary.num_rows matches row count")
    else:
        failed.append(f"summary.num_rows={expected_num_rows} does not match actual rows={len(rows)}")

    unique_cases = {row.get("case_index") for row in rows}
    expected_num_cases = summary.get("num_cases")
    if expected_num_cases is None or int(expected_num_cases) == len(unique_cases):
        passed.append("summary.num_cases matches unique case count")
    else:
        failed.append(f"summary.num_cases={expected_num_cases} does not match unique cases={len(unique_cases)}")

    required = (
        "variant",
        "prompt_index",
        "seed",
        "baseline_image",
        "variant_image",
        "baseline_dit_wall_time_s",
        "variant_dit_wall_time_s",
        "dit_speedup",
        "mse",
        "mae",
        "psnr",
    )
    for index, row in enumerate(rows):
        missing = [key for key in required if key not in row]
        if missing:
            failed.append(f"row {index} missing required keys: {', '.join(missing)}")
            continue
        base_t = row["baseline_dit_wall_time_s"]
        var_t = row["variant_dit_wall_time_s"]
        if not (_finite_number(base_t) and _finite_number(var_t) and base_t > 0 and var_t > 0):
            failed.append(f"row {index} has non-positive or non-finite DiT timing")
        reported_speedup = row["dit_speedup"]
        expected_speedup = float(base_t) / float(var_t) if float(var_t) > 0 else 0.0
        if not _finite_number(reported_speedup) or abs(float(reported_speedup) - expected_speedup) > 1e-4:
            failed.append(f"row {index} dit_speedup does not match baseline/variant DiT timing")
        if row["baseline_image"] == row["variant_image"]:
            failed.append(f"row {index} baseline_image and variant_image are identical")
        if check_images:
            if not _path_exists(row["baseline_image"], metrics_path, repo_root):
                failed.append(f"row {index} baseline_image does not exist: {row['baseline_image']}")
            if not _path_exists(row["variant_image"], metrics_path, repo_root):
                failed.append(f"row {index} variant_image does not exist: {row['variant_image']}")
        for key in ("mse", "mae"):
            if not _finite_number(row[key]):
                failed.append(f"row {index} has non-finite {key}")
        if not _valid_psnr(row["psnr"]):
            failed.append(f"row {index} has invalid psnr")
    if not failed:
        passed.append("all rows pass required integrity checks")
    return passed, failed


def _validate_warmup(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    require_warmup: bool,
) -> tuple[list[str], list[str]]:
    passed: list[str] = []
    failed: list[str] = []
    config = summary.get("benchmark_config")
    if not isinstance(config, dict):
        if require_warmup:
            failed.append("summary.benchmark_config is missing; cannot verify benchmark warmup")
        else:
            passed.append("warmup metadata is not required for this checker run")
        return passed, failed

    try:
        configured_runs = int(config.get("warmup_runs", 0))
    except (TypeError, ValueError):
        configured_runs = -1
    warmup_enabled = bool(config.get("warmup_enabled"))
    warmup_scope = config.get("warmup_scope")

    if configured_runs < 0:
        failed.append("summary.benchmark_config.warmup_runs is not a non-negative integer")
    if require_warmup:
        if not warmup_enabled or configured_runs < 1:
            failed.append("benchmark warmup is disabled or has zero warmup runs")
        else:
            passed.append("benchmark warmup is enabled with at least one run")
        if warmup_scope != "before_each_measured_workflow":
            failed.append("benchmark warmup_scope is not before_each_measured_workflow")
        else:
            passed.append("benchmark warmup is scoped before each measured workflow")
    else:
        passed.append("checker was configured to allow missing or disabled warmup")

    row_warmup_values = [row.get("warmup_runs") for row in rows if "warmup_runs" in row]
    if row_warmup_values:
        parsed: list[int] = []
        for index, value in enumerate(row_warmup_values):
            try:
                parsed.append(int(value))
            except (TypeError, ValueError):
                failed.append(f"row warmup_runs value {index} is not an integer")
        if parsed and any(value != configured_runs for value in parsed):
            failed.append("row warmup_runs values do not match summary benchmark_config")
        elif parsed:
            passed.append("row warmup_runs values match benchmark_config")
        if require_warmup and parsed and any(value < 1 for value in parsed):
            failed.append("at least one row reports zero warmup runs")
    elif require_warmup:
        failed.append("rows do not include warmup_runs metadata")

    row_warmup_enabled = [row.get("warmup_enabled") for row in rows if "warmup_enabled" in row]
    if row_warmup_enabled and any(bool(value) != warmup_enabled for value in row_warmup_enabled):
        failed.append("row warmup_enabled values do not match summary benchmark_config")
    elif row_warmup_enabled:
        passed.append("row warmup_enabled values match benchmark_config")
    elif require_warmup:
        failed.append("rows do not include warmup_enabled metadata")

    return passed, failed


def _variant_result(
    variant: str,
    rows: list[dict[str, Any]],
    benchmark_type: str,
    min_speedup: float,
    min_psnr: float,
) -> dict[str, Any]:
    issues: list[str] = []
    baseline_sum = sum(float(row["baseline_dit_wall_time_s"]) for row in rows)
    variant_sum = sum(float(row["variant_dit_wall_time_s"]) for row in rows)
    speedup = baseline_sum / variant_sum if variant_sum > 0 else 0.0
    saved_ratio = 1.0 - variant_sum / baseline_sum if baseline_sum > 0 else 0.0
    psnrs = [float(row["psnr"]) for row in rows if _valid_psnr(row.get("psnr"))]
    finite_psnrs = [value for value in psnrs if math.isfinite(value)]
    infinite_psnr_rows = sum(1 for value in psnrs if value == float("inf"))
    mean_psnr = _mean(finite_psnrs)
    result: dict[str, Any] = {
        "variant": variant,
        "num_rows": len(rows),
        "case_indices": sorted({row.get("case_index") for row in rows}),
        "baseline_mean_dit_wall_time_s": _mean([float(row["baseline_dit_wall_time_s"]) for row in rows]),
        "variant_mean_dit_wall_time_s": _mean([float(row["variant_dit_wall_time_s"]) for row in rows]),
        "aggregate_dit_speedup": speedup,
        "measured_dit_saved_ratio": saved_ratio,
        "mean_mse": _mean([float(row["mse"]) for row in rows if _finite_number(row.get("mse"))]),
        "mean_mae": _mean([float(row["mae"]) for row in rows if _finite_number(row.get("mae"))]),
        "mean_psnr": mean_psnr,
        "speedup_valid": speedup >= min_speedup,
        "quality_valid": (mean_psnr is None or mean_psnr >= min_psnr),
        "implementation_valid": True,
        "issues": issues,
        "evidence": {
            "infinite_psnr_rows": infinite_psnr_rows,
        },
    }

    if benchmark_type == "cache":
        skipped = sum(int(row.get("variant_skipped_steps", 0) or 0) for row in rows)
        total = sum(int(row.get("variant_total_steps", 0) or 0) for row in rows)
        theoretical = total / (total - skipped) if total > skipped and total > 0 else 1.0
        result["evidence"].update({
            "variant_skipped_steps": skipped,
            "variant_total_steps": total,
            "theoretical_dit_speedup": theoretical,
        })
        if speedup > 1.01 and skipped <= 0:
            issues.append("cache variant reports speedup but skipped zero DiT steps")
        if theoretical > 0 and speedup > theoretical * 1.25:
            issues.append("cache measured speedup is implausibly above skipped-step theoretical speedup")
    elif benchmark_type == "quantization":
        selected = max(int(row.get("selective_quantized_linear_modules", 0) or 0) for row in rows)
        kept = max(int(row.get("selective_kept_high_precision_linear_modules", 0) or 0) for row in rows)
        shapes = max(int(row.get("selective_benchmarked_linear_shapes", 0) or 0) for row in rows)
        result["evidence"].update({
            "selective_benchmarked_linear_shapes": shapes,
            "selective_quantized_linear_modules": selected,
            "selective_kept_high_precision_linear_modules": kept,
        })
        if variant in MASKED_QUANTIZATION_SKILLS:
            issues.append("masked quantization skill appeared as an active benchmark variant")
        if variant.startswith("selective_") and shapes <= 0:
            issues.append("selective quantization variant did not report benchmarked Linear shapes")
    elif benchmark_type == "sparse_attention":
        kernel_calls = sum(int(row.get("official_kernel_calls", 0) or 0) for row in rows)
        attention_calls = sum(int(row.get("attention_calls", 0) or 0) for row in rows)
        fallback_calls = sum(int(row.get("fallback_attention_calls", 0) or 0) for row in rows)
        skipped_fraction = _mean([float(row.get("mean_skipped_attention_fraction", 0.0) or 0.0) for row in rows])
        result["evidence"].update({
            "official_kernel_calls": kernel_calls,
            "attention_calls": attention_calls,
            "fallback_attention_calls": fallback_calls,
            "mean_skipped_attention_fraction": skipped_fraction,
        })
        if kernel_calls <= 0:
            issues.append("sparse attention variant did not call the official GPU kernel")
        if attention_calls > 0 and fallback_calls >= attention_calls and speedup > 1.01:
            issues.append("sparse speedup claimed but all attention calls fell back to dense")

    if issues:
        result["implementation_valid"] = False
    return result


def build_checker_result(
    metrics_path: Path,
    output_path: Path,
    benchmark_type: str,
    min_speedup: float,
    min_psnr: float,
    check_images: bool,
    repo_root: Path | None,
    diff_report_path: Path | None,
    static_audit_status: str,
    static_audit_notes: list[str],
    require_warmup: bool,
) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    summary = metrics.get("summary", {})
    rows = metrics.get("rows", [])
    if benchmark_type == "auto":
        benchmark_type = _detect_benchmark_type(rows, summary)

    passed, failed = _validate_rows(rows, summary, metrics_path, repo_root, check_images)
    warmup_passed, warmup_failed = _validate_warmup(rows, summary, require_warmup)
    passed.extend(warmup_passed)
    failed.extend(warmup_failed)
    variant_results = [
        _variant_result(variant, variant_rows, benchmark_type, min_speedup, min_psnr)
        for variant, variant_rows in sorted(_group_rows(rows).items())
    ]
    implementation_issues = [
        f"{item['variant']}: {issue}"
        for item in variant_results
        for issue in item["issues"]
    ]

    unsupported = summary.get("unsupported") or {}
    if unsupported:
        implementation_issues.append(f"unsupported variants present: {', '.join(sorted(unsupported))}")

    active_variants = {item["variant"] for item in variant_results}
    masked_active = sorted(active_variants & MASKED_QUANTIZATION_SKILLS)
    if masked_active:
        implementation_issues.append(f"masked quantization variants were active: {', '.join(masked_active)}")
    if static_audit_status == "fail":
        implementation_issues.append("static source audit failed")

    diff_report = None
    if diff_report_path is not None and diff_report_path.exists():
        diff_report = _load_json(diff_report_path)

    benchmark_valid = not failed
    implementation_valid = benchmark_valid and not implementation_issues
    any_speedup = any(item["speedup_valid"] for item in variant_results)
    all_quality_valid = all(item["quality_valid"] for item in variant_results)
    if not benchmark_valid:
        recommendation = "rerun"
    elif not implementation_valid:
        recommendation = "reject"
    elif any_speedup and all_quality_valid:
        recommendation = "accept"
    elif any_speedup:
        recommendation = "needs_review"
    else:
        recommendation = "iterate"

    notes: list[str] = []
    if recommendation == "accept":
        notes.append("At least one variant passed speed and quality thresholds with valid benchmark structure.")
    elif recommendation == "iterate":
        notes.append("Benchmark structure is valid, but no variant passed the configured speedup threshold.")
    elif recommendation == "needs_review":
        notes.append("A speedup was measured, but quality threshold or other review criteria need human judgment.")
    elif recommendation == "reject":
        notes.append("Implementation evidence failed integrity checks; inspect variant issues before another run.")
    else:
        notes.append("Benchmark output failed structural validation; rerun after fixing benchmark or artifact paths.")

    return {
        "schema_version": SCHEMA_VERSION,
        "role": "checker",
        "created_at_utc": _now(),
        "inputs": {
            "metrics_path": str(metrics_path),
            "output_path": str(output_path),
            "benchmark_type": benchmark_type,
            "min_speedup": min_speedup,
            "min_psnr": min_psnr,
            "check_images": check_images,
            "diff_report_path": str(diff_report_path) if diff_report_path else None,
            "static_audit_status": static_audit_status,
            "require_warmup": require_warmup,
        },
        "benchmark_valid": benchmark_valid,
        "implementation_valid": implementation_valid,
        "recommendation": recommendation,
        "integrity_checks": {
            "passed": passed,
            "failed": failed,
        },
        "implementation_issues": implementation_issues,
        "static_audit": {
            "status": static_audit_status,
            "notes": static_audit_notes,
        },
        "summary": {
            "num_cases": summary.get("num_cases"),
            "num_rows": summary.get("num_rows", len(rows)),
            "variants": sorted(active_variants),
            "unsupported": unsupported,
            "masked_skills": summary.get("masked_skills", []),
            "benchmark_config": summary.get("benchmark_config", {}),
        },
        "variant_results": variant_results,
        "diff_report": diff_report,
        "next_reviewer_notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--benchmark-type", default="auto", choices=["auto", "cache", "quantization", "sparse_attention", "generic"])
    parser.add_argument("--min-speedup", type=float, default=1.01)
    parser.add_argument("--min-psnr", type=float, default=0.0)
    parser.add_argument("--no-image-check", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--diff-report", type=Path, default=None)
    parser.add_argument("--static-audit-status", default="not_run", choices=["pass", "fail", "not_run"])
    parser.add_argument("--static-audit-note", action="append", default=[])
    parser.add_argument("--allow-no-warmup", action="store_true")
    args = parser.parse_args()

    metrics_path = args.metrics.resolve()
    output_path = args.output.resolve() if args.output else metrics_path.parent / "checker_result.json"
    result = build_checker_result(
        metrics_path=metrics_path,
        output_path=output_path,
        benchmark_type=args.benchmark_type,
        min_speedup=args.min_speedup,
        min_psnr=args.min_psnr,
        check_images=not args.no_image_check,
        repo_root=args.repo_root.resolve() if args.repo_root else None,
        diff_report_path=args.diff_report.resolve() if args.diff_report else None,
        static_audit_status=args.static_audit_status,
        static_audit_notes=args.static_audit_note,
        require_warmup=not args.allow_no_warmup,
    )
    _write_json(output_path, result)
    print(output_path)


if __name__ == "__main__":
    main()
