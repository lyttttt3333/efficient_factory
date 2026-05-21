from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draft_canvas.flux_schnell_cache_canvas import build_flux_schnell_baseline
from draft_canvas.flux_schnell_candidate_canvas import (
    CANDIDATE_METADATA,
    CANDIDATE_NAME,
    build_flux_schnell_candidate,
    clear_candidate_state,
    prepare_candidate,
)
from model.download_flux_schnell import download_flux_schnell_split
from model.flux_schnell import FLUX_SCHNELL_SPLIT


PROMPTS = [
    "a small red robot reading a book under a desk lamp, detailed product photo",
    "a quiet mountain lake at sunrise with mist, cinematic landscape",
    "a glass teapot filled with blue flowers on a wooden table, studio lighting",
    "a futuristic city tram station in the rain, realistic concept art",
]

SEEDS = [101, 202, 303, 404]

DEFAULT_MAX_CASES = 1
MAX_SEQUENCE_LENGTH_NOTE = (
    "CLIPTextEncodeFlux in this ComfyUI checkout does not expose max_sequence_length; "
    "the benchmark records the requested official value for config traceability."
)
TIME_SHIFT_NOTE = "ComfyUI BasicScheduler handles time shift internally; no explicit canvas input is exposed."


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _runtime_imports():
    from efficient_skill.common.local_inference import make_runner, run_workflow, setup_local_runtime

    return make_runner, run_workflow, setup_local_runtime


def _compare_images(baseline_image: Path, candidate_image: Path) -> dict[str, float]:
    from efficient_skill.common.image_metrics import compare_images

    return compare_images(baseline_image, candidate_image)


def _clear_cuda_and_model_caches() -> None:
    import gc

    import torch
    import comfy.model_management

    try:
        comfy.model_management.unload_all_models()
    except RuntimeError as exc:
        if "Attempted to set the storage of a tensor" not in str(exc):
            raise
        comfy.model_management.current_loaded_models.clear()
    comfy.model_management.soft_empty_cache(force=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _release_loaded_models(runner) -> None:
    runner.literal_cache.clear()
    clear_candidate_state()
    _clear_cuda_and_model_caches()


def _run_measured_workflow(
    runner,
    workflow_builder,
    output_root: Path,
    warmup_runs: int,
):
    _, run_workflow, _ = _runtime_imports()
    _release_loaded_models(runner)
    for warmup_index in range(warmup_runs):
        run_workflow(runner, workflow_builder(f"warmup_{warmup_index:02d}"), output_root)
        _release_loaded_models(runner)
    _release_loaded_models(runner)
    result = run_workflow(runner, workflow_builder("measure"), output_root)
    _release_loaded_models(runner)
    return result


def _benchmark_cases(max_cases: int | None) -> list[tuple[int, int, str, int]]:
    cases = [(p_i, s_i, prompt, seed) for p_i, prompt in enumerate(PROMPTS) for s_i, seed in enumerate(SEEDS)]
    if max_cases is not None:
        cases = cases[:max_cases]
    return cases


def _common_config(
    max_cases: int | None,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    max_sequence_length: int,
    sampler_name: str,
    scheduler: str,
    warmup_runs: int,
) -> dict[str, Any]:
    return {
        "case_limit": max_cases,
        "available_cases": len(PROMPTS) * len(SEEDS),
        "prompts": list(PROMPTS),
        "seeds": list(SEEDS),
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance_scale,
        "guidance_scale": guidance_scale,
        "max_sequence_length": max_sequence_length,
        "max_sequence_length_applied": False,
        "max_sequence_length_note": MAX_SEQUENCE_LENGTH_NOTE,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
        "time_shift": "not_exposed_by_basic_scheduler",
        "time_shift_note": TIME_SHIFT_NOTE,
        "warmup_enabled": warmup_runs > 0,
        "warmup_runs": warmup_runs,
        "warmup_scope": "before_each_measured_workflow",
        "timing_scope": "dit_denoise_only",
    }


def _configs_match(current: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        current_value = current.get(key)
        if isinstance(expected_value, float):
            try:
                if abs(float(current_value) - expected_value) > 1e-9:
                    return False
            except (TypeError, ValueError):
                return False
        elif current_value != expected_value:
            return False
    return True


def _baseline_rows_by_case(
    rows: list[dict[str, Any]],
    cases: list[tuple[int, int, str, int]],
) -> dict[int, dict[str, Any]] | None:
    if len(rows) < len(cases):
        return None
    by_case: dict[int, dict[str, Any]] = {}
    for case_index, (prompt_index, seed_index, prompt, seed) in enumerate(cases):
        matching = [
            row
            for row in rows
            if row.get("case_index") == case_index
            and row.get("prompt_index") == prompt_index
            and row.get("seed_index") == seed_index
            and row.get("prompt") == prompt
            and row.get("seed") == seed
        ]
        if len(matching) != 1:
            return None
        row = matching[0]
        image_path = Path(str(row.get("baseline_image", "")))
        if not image_path.exists():
            return None
        if not isinstance(row.get("baseline_dit_wall_time_s"), (int, float)) or row["baseline_dit_wall_time_s"] <= 0:
            return None
        by_case[case_index] = row
    return by_case


def _load_fixed_baseline(
    artifact_root: Path,
    expected_config: dict[str, Any],
    cases: list[tuple[int, int, str, int]],
) -> dict[int, dict[str, Any]] | None:
    metrics_path = artifact_root / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    summary = metrics.get("summary", {})
    rows = metrics.get("rows", [])
    if not isinstance(summary, dict) or not isinstance(rows, list):
        return None
    if summary.get("artifact_role") != "fixed_baseline":
        return None
    config = summary.get("benchmark_config")
    if not isinstance(config, dict) or not _configs_match(config, expected_config):
        return None
    return _baseline_rows_by_case(rows, cases)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    first_fields = list(rows[0].keys())
    extra_fields = sorted({key for row in rows for key in row.keys()} - set(first_fields))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=first_fields + extra_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_fixed_baseline_summary(
    artifact_root: Path,
    rows: list[dict[str, Any]],
    benchmark_config: dict[str, Any],
) -> None:
    summary = {
        "artifact_role": "fixed_baseline",
        "num_cases": len({r["case_index"] for r in rows}),
        "num_rows": len(rows),
        "benchmark_config": benchmark_config,
        "baseline_mean_dit_wall_time_s": _mean([r["baseline_dit_wall_time_s"] for r in rows]),
        "baseline_mean_end_to_end_wall_time_s": _mean([r["baseline_end_to_end_wall_time_s"] for r in rows]),
    }
    (artifact_root / "metrics.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    _write_csv(artifact_root / "metrics.csv", rows)

    lines = [
        "# Flux Schnell Fixed Baseline",
        "",
        f"- cases: {summary['num_cases']}",
        f"- width: {benchmark_config['width']}",
        f"- height: {benchmark_config['height']}",
        f"- steps: {benchmark_config['steps']}",
        f"- guidance_scale: {benchmark_config['guidance_scale']}",
        f"- max_sequence_length: {benchmark_config['max_sequence_length']} (recorded only)",
        f"- sampler: {benchmark_config['sampler_name']}",
        f"- scheduler: {benchmark_config['scheduler']}",
        f"- warmup runs: {benchmark_config['warmup_runs']} before each measured workflow",
        "- timing: DiT denoise only; text encoder, VAE decode, and image save are excluded from speedup.",
        "",
        "| case | prompt_id | seed | baseline_dit_s | end_to_end_s | image |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_index']} | {row['prompt_index']} | {row['seed']} | "
            f"{row['baseline_dit_wall_time_s']:.4f} | "
            f"{row['baseline_end_to_end_wall_time_s']:.4f} | {row['baseline_image']} |"
        )
    (artifact_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _create_fixed_baseline(
    artifact_root: Path,
    output_root: Path,
    cases: list[tuple[int, int, str, int]],
    benchmark_config: dict[str, Any],
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    sampler_name: str,
    scheduler: str,
    warmup_runs: int,
) -> dict[int, dict[str, Any]]:
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    output_root.mkdir(parents=True, exist_ok=True)
    make_runner, _, setup_local_runtime = _runtime_imports()
    setup_local_runtime(output_root)
    runner = make_runner()
    rows: list[dict[str, Any]] = []
    try:
        for case_index, (prompt_index, seed_index, prompt, seed) in enumerate(cases):
            baseline_result = _run_measured_workflow(
                runner,
                lambda phase, prompt=prompt, seed=seed, case_index=case_index, prompt_index=prompt_index, seed_index=seed_index: build_flux_schnell_baseline(
                    prompt=prompt,
                    seed=seed,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance=guidance_scale,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    filename_prefix=f"{phase}/baseline/case_{case_index:03d}_p{prompt_index}_s{seed_index}",
                ),
                output_root,
                warmup_runs,
            )
            rows.append({
                "case_index": case_index,
                "prompt_index": prompt_index,
                "seed_index": seed_index,
                "seed": seed,
                "prompt": prompt,
                "baseline_image": str(baseline_result.image_path),
                "baseline_dit_wall_time_s": baseline_result.denoise_wall_time_s,
                "baseline_end_to_end_wall_time_s": baseline_result.wall_time_s,
                "baseline_total_steps": baseline_result.total_steps,
                "baseline_skipped_steps": baseline_result.skipped_steps,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "max_sequence_length": benchmark_config["max_sequence_length"],
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "warmup_enabled": warmup_runs > 0,
                "warmup_runs": warmup_runs,
            })
            write_fixed_baseline_summary(artifact_root, rows, benchmark_config)
    finally:
        _release_loaded_models(runner)
    return {row["case_index"]: row for row in rows}


def _ensure_fixed_baseline(
    repo_root: Path,
    baseline_artifact_name: str,
    cases: list[tuple[int, int, str, int]],
    benchmark_config: dict[str, Any],
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    sampler_name: str,
    scheduler: str,
    warmup_runs: int,
) -> tuple[Path, dict[int, dict[str, Any]], str]:
    artifact_root = repo_root / "benchmark" / "artifacts" / baseline_artifact_name
    rows_by_case = _load_fixed_baseline(artifact_root, benchmark_config, cases)
    if rows_by_case is not None:
        return artifact_root, rows_by_case, "reused"
    rows_by_case = _create_fixed_baseline(
        artifact_root=artifact_root,
        output_root=artifact_root / "outputs",
        cases=cases,
        benchmark_config=benchmark_config,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        sampler_name=sampler_name,
        scheduler=scheduler,
        warmup_runs=warmup_runs,
    )
    return artifact_root, rows_by_case, "created"


def write_summary(
    artifact_root: Path,
    rows: list[dict[str, Any]],
    benchmark_config: dict[str, Any],
) -> None:
    finite_psnrs = [row["psnr"] for row in rows if row["psnr"] != float("inf")]
    baseline_dit_sum = sum(row["baseline_dit_wall_time_s"] for row in rows)
    candidate_dit_sum = sum(row["candidate_dit_wall_time_s"] for row in rows)
    measured_speedup = baseline_dit_sum / candidate_dit_sum if candidate_dit_sum > 0 else 0.0
    measured_saved_ratio = 1.0 - (candidate_dit_sum / baseline_dit_sum) if baseline_dit_sum > 0 else 0.0
    candidate_stats = {
        "num_cases": len(rows),
        "baseline_mean_dit_wall_time_s": _mean([row["baseline_dit_wall_time_s"] for row in rows]),
        "variant_mean_dit_wall_time_s": _mean([row["candidate_dit_wall_time_s"] for row in rows]),
        "aggregate_dit_speedup": measured_speedup,
        "measured_dit_saved_ratio": measured_saved_ratio,
        "baseline_mean_end_to_end_wall_time_s": _mean([row["baseline_end_to_end_wall_time_s"] for row in rows]),
        "variant_mean_end_to_end_wall_time_s": _mean([row["candidate_end_to_end_wall_time_s"] for row in rows]),
        "mean_mse": _mean([row["mse"] for row in rows]),
        "mean_mae": _mean([row["mae"] for row in rows]),
        "mean_psnr": _mean(finite_psnrs) if finite_psnrs else float("inf"),
    }
    summary = {
        "artifact_role": "candidate_comparison",
        "num_cases": len({row["case_index"] for row in rows}),
        "num_rows": len(rows),
        "benchmark_config": benchmark_config,
        "candidate_name": CANDIDATE_NAME,
        "candidate_metadata": CANDIDATE_METADATA,
        "variants": {
            CANDIDATE_NAME: candidate_stats,
        },
    }
    (artifact_root / "metrics.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    _write_csv(artifact_root / "metrics.csv", rows)

    lines = [
        "# Flux Schnell Candidate Benchmark",
        "",
        f"- cases: {summary['num_cases']}",
        f"- candidate: {CANDIDATE_NAME}",
        f"- baseline scope: {benchmark_config.get('baseline_scope', 'per_iteration_inline')}",
        f"- baseline artifact: {benchmark_config.get('baseline_artifact_name') or 'none'}",
        f"- baseline mode: {benchmark_config.get('baseline_reuse_mode', 'inline')}",
        f"- width: {benchmark_config['width']}",
        f"- height: {benchmark_config['height']}",
        f"- steps: {benchmark_config['steps']}",
        f"- guidance_scale: {benchmark_config['guidance_scale']}",
        f"- max_sequence_length: {benchmark_config['max_sequence_length']} (recorded only)",
        f"- sampler: {benchmark_config['sampler_name']}",
        f"- scheduler: {benchmark_config['scheduler']}",
        f"- warmup runs: {benchmark_config['warmup_runs']} before each measured workflow",
        "- timing: DiT denoise only; text encoder, VAE decode, image save, candidate preparation, and compile overhead are excluded from speedup.",
        "",
        "| candidate | cases | baseline_dit_s | candidate_dit_s | saved% | speedup | mse | mae | psnr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {CANDIDATE_NAME} | {candidate_stats['num_cases']} | "
            f"{candidate_stats['baseline_mean_dit_wall_time_s']:.4f} | "
            f"{candidate_stats['variant_mean_dit_wall_time_s']:.4f} | "
            f"{candidate_stats['measured_dit_saved_ratio'] * 100:.2f}% | "
            f"{candidate_stats['aggregate_dit_speedup']:.4f} | "
            f"{candidate_stats['mean_mse']:.8f} | {candidate_stats['mean_mae']:.8f} | "
            f"{candidate_stats['mean_psnr']:.4f} |"
        ),
    ]
    if rows:
        lines.extend([
            "",
            "| case | prompt_id | seed | baseline_dit_s | candidate_dit_s | saved% | speedup | mse | mae | psnr |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            lines.append(
                f"| {row['case_index']} | {row['prompt_index']} | {row['seed']} | "
                f"{row['baseline_dit_wall_time_s']:.4f} | {row['candidate_dit_wall_time_s']:.4f} | "
                f"{row['dit_saved_ratio'] * 100:.2f}% | {row['dit_speedup']:.4f} | "
                f"{row['mse']:.8f} | {row['mae']:.8f} | {row['psnr']:.4f} |"
            )
    (artifact_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    repo_root: Path,
    max_cases: int | None,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    max_sequence_length: int,
    sampler_name: str,
    scheduler: str,
    ensure_model: bool,
    warmup_runs: int,
    artifact_name: str,
    baseline_artifact_name: str | None,
) -> Path:
    if max_cases is not None and max_cases < 1:
        raise ValueError("--max-cases must be >= 1")
    if warmup_runs < 0:
        raise ValueError("--warmup-runs must be >= 0")
    if baseline_artifact_name and baseline_artifact_name == artifact_name:
        raise ValueError("--baseline-artifact-name must differ from --artifact-name")

    artifact_root = repo_root / "benchmark" / "artifacts" / artifact_name
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    output_root = artifact_root / "outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    if ensure_model:
        download_flux_schnell_split(repo_root)
    missing = [
        component.local_path(repo_root)
        for component in FLUX_SCHNELL_SPLIT.components()
        if not component.local_path(repo_root).exists()
    ]
    if missing:
        missing_list = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing Flux split model files:\n{missing_list}")

    cases = _benchmark_cases(max_cases)
    fixed_config = _common_config(
        max_cases=max_cases,
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
        sampler_name=sampler_name,
        scheduler=scheduler,
        warmup_runs=warmup_runs,
    )
    baseline_root: Path | None = None
    baseline_rows_by_case: dict[int, dict[str, Any]] | None = None
    baseline_reuse_mode = "inline_per_iteration"
    if baseline_artifact_name:
        baseline_root, baseline_rows_by_case, baseline_reuse_mode = _ensure_fixed_baseline(
            repo_root=repo_root,
            baseline_artifact_name=baseline_artifact_name,
            cases=cases,
            benchmark_config={**fixed_config, "artifact_role": "fixed_baseline"},
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            sampler_name=sampler_name,
            scheduler=scheduler,
            warmup_runs=warmup_runs,
        )

    make_runner, _, setup_local_runtime = _runtime_imports()
    setup_local_runtime(output_root)
    runner = make_runner()
    rows: list[dict[str, Any]] = []
    candidate_prepare_result = prepare_candidate(
        width=width,
        height=height,
        steps=steps,
        guidance_scale=guidance_scale,
        max_sequence_length=max_sequence_length,
        sampler_name=sampler_name,
        scheduler=scheduler,
    )
    benchmark_config = {
        **fixed_config,
        "artifact_role": "candidate_comparison",
        "baseline_scope": "run_level_fixed_baseline" if baseline_artifact_name else "per_iteration_inline",
        "baseline_artifact_name": baseline_artifact_name,
        "baseline_artifact_path": str(baseline_root) if baseline_root is not None else None,
        "baseline_reuse_mode": baseline_reuse_mode,
        "candidate_name": CANDIDATE_NAME,
        "candidate_metadata": CANDIDATE_METADATA,
        "candidate_prepare_result": candidate_prepare_result,
        "candidate_interface": "draft_canvas.flux_schnell_candidate_canvas.build_flux_schnell_candidate",
        "benchmark_interface": "benchmark/flux_schnell_benchmark.py",
        "memory_cleanup": "release_model_cache_and_candidate_state_before_and_after_warmup_and_measure",
    }

    try:
        for case_index, (prompt_index, seed_index, prompt, seed) in enumerate(cases):
            if baseline_rows_by_case is None:
                baseline_result = _run_measured_workflow(
                    runner,
                    lambda phase, prompt=prompt, seed=seed, case_index=case_index, prompt_index=prompt_index, seed_index=seed_index: build_flux_schnell_baseline(
                        prompt=prompt,
                        seed=seed,
                        width=width,
                        height=height,
                        steps=steps,
                        guidance=guidance_scale,
                        sampler_name=sampler_name,
                        scheduler=scheduler,
                        filename_prefix=f"{phase}/baseline/case_{case_index:03d}_p{prompt_index}_s{seed_index}",
                    ),
                    output_root,
                    warmup_runs,
                )
                baseline_image = baseline_result.image_path
                baseline_dit_wall_time_s = baseline_result.denoise_wall_time_s
                baseline_end_to_end_wall_time_s = baseline_result.wall_time_s
                baseline_total_steps = baseline_result.total_steps
                baseline_skipped_steps = baseline_result.skipped_steps
            else:
                baseline_row = baseline_rows_by_case[case_index]
                baseline_image = Path(str(baseline_row["baseline_image"]))
                baseline_dit_wall_time_s = baseline_row["baseline_dit_wall_time_s"]
                baseline_end_to_end_wall_time_s = baseline_row["baseline_end_to_end_wall_time_s"]
                baseline_total_steps = baseline_row.get("baseline_total_steps", 0)
                baseline_skipped_steps = baseline_row.get("baseline_skipped_steps", 0)

            candidate_result = _run_measured_workflow(
                runner,
                lambda phase, prompt=prompt, seed=seed, case_index=case_index, prompt_index=prompt_index, seed_index=seed_index: build_flux_schnell_candidate(
                    prompt=prompt,
                    seed=seed,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance=guidance_scale,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    filename_prefix=f"{phase}/candidate/case_{case_index:03d}_p{prompt_index}_s{seed_index}",
                ),
                output_root,
                warmup_runs,
            )
            metrics = _compare_images(baseline_image, candidate_result.image_path)
            dit_speedup = (
                baseline_dit_wall_time_s / candidate_result.denoise_wall_time_s
                if candidate_result.denoise_wall_time_s > 0
                else 0.0
            )
            dit_saved_ratio = (
                1.0 - (candidate_result.denoise_wall_time_s / baseline_dit_wall_time_s)
                if baseline_dit_wall_time_s > 0
                else 0.0
            )
            rows.append({
                "case_index": case_index,
                "variant": CANDIDATE_NAME,
                "candidate_name": CANDIDATE_NAME,
                "prompt_index": prompt_index,
                "seed_index": seed_index,
                "seed": seed,
                "prompt": prompt,
                "baseline_image": str(baseline_image),
                "variant_image": str(candidate_result.image_path),
                "candidate_image": str(candidate_result.image_path),
                "baseline_dit_wall_time_s": baseline_dit_wall_time_s,
                "variant_dit_wall_time_s": candidate_result.denoise_wall_time_s,
                "candidate_dit_wall_time_s": candidate_result.denoise_wall_time_s,
                "dit_speedup": dit_speedup,
                "dit_saved_ratio": dit_saved_ratio,
                "baseline_total_steps": baseline_total_steps,
                "baseline_skipped_steps": baseline_skipped_steps,
                "baseline_end_to_end_wall_time_s": baseline_end_to_end_wall_time_s,
                "variant_end_to_end_wall_time_s": candidate_result.wall_time_s,
                "candidate_end_to_end_wall_time_s": candidate_result.wall_time_s,
                "warmup_enabled": warmup_runs > 0,
                "warmup_runs": warmup_runs,
                "benchmark_baseline_source": benchmark_config["baseline_scope"],
                "baseline_artifact_name": baseline_artifact_name,
                "baseline_artifact_path": str(baseline_root) if baseline_root is not None else None,
                "baseline_reuse_mode": baseline_reuse_mode,
                "width": width,
                "height": height,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "max_sequence_length": max_sequence_length,
                "max_sequence_length_applied": False,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "time_shift": "not_exposed_by_basic_scheduler",
                **metrics,
            })
            write_summary(artifact_root, rows, benchmark_config)
    finally:
        _release_loaded_models(runner)
    write_summary(artifact_root, rows, benchmark_config)
    return artifact_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cases", type=int, default=DEFAULT_MAX_CASES)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--sampler-name", default="euler")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--artifact-name", default="flux_schnell_candidate_benchmark")
    parser.add_argument("--baseline-artifact-name", default=None)
    args = parser.parse_args()

    artifact_root = run_benchmark(
        repo_root=repo_root_from_here(),
        max_cases=args.max_cases,
        width=args.width,
        height=args.height,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        sampler_name=args.sampler_name,
        scheduler=args.scheduler,
        ensure_model=not args.no_download,
        warmup_runs=0 if args.no_warmup else args.warmup_runs,
        artifact_name=args.artifact_name,
        baseline_artifact_name=args.baseline_artifact_name,
    )
    print(artifact_root)


if __name__ == "__main__":
    main()
