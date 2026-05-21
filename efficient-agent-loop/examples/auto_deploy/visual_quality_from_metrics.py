from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "auto_deploy.visual_quality.v1"
QUALITY_LABELS = {
    "better",
    "similar",
    "minor_regression",
    "major_regression",
    "inconclusive",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--max-pairs", type=int, default=1)
    parser.add_argument("--model", default=os.environ.get("EAL_VISUAL_REVIEW_MODEL", ""))
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.environ.get("EAL_VISUAL_REVIEW_TIMEOUT_SECONDS", "240")),
    )
    parser.add_argument("--optional", action="store_true")
    args = parser.parse_args()

    result = build_visual_quality_result(
        metrics_path=args.metrics.resolve(),
        output_path=args.output.resolve(),
        repo_root=args.repo_root.resolve(),
        max_pairs=args.max_pairs,
        model=args.model.strip(),
        timeout_seconds=args.timeout_seconds,
        required=not args.optional,
    )
    _write_json(args.output, result)
    print(args.output)
    return 0


def build_visual_quality_result(
    *,
    metrics_path: Path,
    output_path: Path,
    repo_root: Path,
    max_pairs: int,
    model: str,
    timeout_seconds: float,
    required: bool,
) -> dict[str, Any]:
    rows = _select_rows(metrics_path, max_pairs)
    base = {
        "schema_version": SCHEMA_VERSION,
        "status": "pending",
        "required": required,
        "evaluator": "codex",
        "model": model or None,
        "metrics_path": str(metrics_path),
        "pairs": [],
        "overall": {
            "qualitative_pass": None,
            "quality_label": "inconclusive",
            "summary": "",
            "recommendation": "needs_review",
        },
    }
    if not rows:
        base["status"] = "skipped"
        base["overall"]["summary"] = "No image rows were available in metrics.json."
        return base

    if os.environ.get("EAL_ENABLE_NESTED_CODEX_VISUAL", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        return _local_visual_proxy(base, rows, required)

    codex = shutil.which("codex")
    if codex is None:
        base["status"] = "failed"
        base["overall"]["summary"] = "Codex CLI was not found, so multimodal image review could not run."
        base["overall"]["recommendation"] = "rerun" if required else "needs_review"
        return base

    schema_path = output_path.with_name(output_path.stem + "_schema.json")
    _write_json(schema_path, _output_schema())

    pair_results = []
    for index, row in enumerate(rows):
        pair_results.append(
            _review_pair(
                codex=codex,
                row=row,
                pair_index=index,
                output_path=output_path,
                schema_path=schema_path,
                repo_root=repo_root,
                model=model,
                timeout_seconds=timeout_seconds,
            )
        )

    base["pairs"] = pair_results
    failed = [item for item in pair_results if item.get("status") != "completed"]
    completed = [item for item in pair_results if item.get("status") == "completed"]
    qualitative_failures = [
        item
        for item in completed
        if item.get("qualitative_pass") is False
        or item.get("quality_label") == "major_regression"
    ]

    if failed:
        base["status"] = "failed" if required else "completed"
        base["overall"]["qualitative_pass"] = False if required else None
        base["overall"]["quality_label"] = "inconclusive"
        base["overall"]["summary"] = (
            f"{len(failed)} visual review pair(s) failed; see pair logs for details."
        )
        base["overall"]["recommendation"] = "rerun" if required else "needs_review"
    elif qualitative_failures:
        base["status"] = "completed"
        base["overall"]["qualitative_pass"] = False
        base["overall"]["quality_label"] = "major_regression"
        base["overall"]["summary"] = "At least one candidate image has a qualitative regression."
        base["overall"]["recommendation"] = "needs_fix"
    else:
        labels = {str(item.get("quality_label", "inconclusive")) for item in completed}
        base["status"] = "completed"
        base["overall"]["qualitative_pass"] = bool(completed)
        base["overall"]["quality_label"] = _overall_label(labels)
        base["overall"]["summary"] = "All reviewed image pairs passed qualitative visual inspection."
        base["overall"]["recommendation"] = "visual_pass"
        base["overall"]["recommendation_scope"] = "visual_quality_only"
    return base


def _local_visual_proxy(base: dict[str, Any], rows: list[dict[str, Any]], required: bool) -> dict[str, Any]:
    pair_results = []
    for index, row in enumerate(rows):
        psnr = _float_or_none(row.get("psnr"))
        mae = _float_or_none(row.get("mae"))
        mse = _float_or_none(row.get("mse"))
        label = "similar"
        qualitative_pass = True
        visible_regressions: list[str] = []
        if psnr is None:
            label = "inconclusive"
            qualitative_pass = False
            visible_regressions.append("PSNR was unavailable for local visual proxy.")
        elif psnr < 18.0 or (mae is not None and mae > 0.10):
            label = "major_regression"
            qualitative_pass = False
            visible_regressions.append(
                "Large pixel-level drift indicates a likely visible or semantic regression."
            )
        elif psnr < 20.0 or (mae is not None and mae > 0.06):
            label = "minor_regression"
            qualitative_pass = False
            visible_regressions.append(
                "Moderate pixel-level drift is too large for automatic acceptance."
            )
        pair_results.append(
            {
                "case_index": row.get("case_index"),
                "variant": row.get("variant"),
                "prompt": row.get("prompt"),
                "baseline_image": str(row.get("baseline_image", "")),
                "variant_image": str(row.get("variant_image", "")),
                "status": "completed",
                "qualitative_pass": qualitative_pass,
                "quality_label": label,
                "baseline_summary": "Baseline image path recorded; nested Codex visual review disabled.",
                "variant_summary": "Candidate image path recorded; local metric proxy used.",
                "artifact_summary": (
                    f"Local proxy from PSNR={psnr}, MAE={mae}, MSE={mse}."
                ),
                "prompt_alignment": "Not directly assessed by local proxy.",
                "visible_regressions": visible_regressions,
                "confidence": "medium" if psnr is not None else "low",
                "evaluator": "local_image_metric_proxy",
                "numeric": {
                    "mse": mse,
                    "mae": mae,
                    "psnr": psnr,
                    "baseline_dit_wall_time_s": row.get("baseline_dit_wall_time_s"),
                    "variant_dit_wall_time_s": row.get("variant_dit_wall_time_s"),
                    "dit_speedup": row.get("dit_speedup"),
                },
                "pair_index": index,
            }
        )

    failures = [item for item in pair_results if not item.get("qualitative_pass")]
    base["evaluator"] = "local_image_metric_proxy"
    base["status"] = "completed"
    base["pairs"] = pair_results
    if failures:
        worst_label = "major_regression" if any(
            item.get("quality_label") == "major_regression" for item in failures
        ) else "minor_regression"
        base["overall"] = {
            "qualitative_pass": False,
            "quality_label": worst_label,
            "summary": (
                "Local qualitative proxy flagged candidate image drift; nested Codex "
                "multimodal review is disabled for stable unattended runs."
            ),
            "recommendation": "needs_fix" if required else "needs_review",
        }
    else:
        base["overall"] = {
            "qualitative_pass": True,
            "quality_label": "similar",
            "summary": (
                "Local qualitative proxy found no large image-metric regression; "
                "nested Codex multimodal review is disabled for stable unattended runs."
            ),
            "recommendation": "visual_pass",
            "recommendation_scope": "visual_quality_only",
        }
    return base


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _select_rows(metrics_path: Path, max_pairs: int) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = metrics.get("rows", [])
    if not isinstance(rows, list):
        return []
    selected = []
    seen_variants = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        variant = str(row.get("variant", "unknown"))
        if variant in seen_variants and len(selected) >= max_pairs:
            continue
        if not row.get("baseline_image") or not row.get("variant_image"):
            continue
        selected.append(row)
        seen_variants.add(variant)
        if len(selected) >= max_pairs:
            break
    return selected


def _review_pair(
    *,
    codex: str,
    row: dict[str, Any],
    pair_index: int,
    output_path: Path,
    schema_path: Path,
    repo_root: Path,
    model: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    baseline_image = Path(str(row["baseline_image"]))
    variant_image = Path(str(row["variant_image"]))
    pair_base = {
        "case_index": row.get("case_index"),
        "variant": row.get("variant"),
        "prompt": row.get("prompt"),
        "baseline_image": str(baseline_image),
        "variant_image": str(variant_image),
        "numeric": {
            "mse": row.get("mse"),
            "mae": row.get("mae"),
            "psnr": row.get("psnr"),
            "baseline_dit_wall_time_s": row.get("baseline_dit_wall_time_s"),
            "variant_dit_wall_time_s": row.get("variant_dit_wall_time_s"),
            "dit_speedup": row.get("dit_speedup"),
        },
    }
    if not baseline_image.exists() or not variant_image.exists():
        return {
            **pair_base,
            "status": "failed",
            "error": "baseline_image or variant_image does not exist.",
        }

    last_message = output_path.with_name(f"{output_path.stem}_pair_{pair_index:03d}.json")
    stdout_path = output_path.with_name(f"{output_path.stem}_pair_{pair_index:03d}.stdout.jsonl")
    stderr_path = output_path.with_name(f"{output_path.stem}_pair_{pair_index:03d}.stderr.txt")
    command = [
        codex,
        "exec",
        "--json",
        "--full-auto",
        "--sandbox",
        "read-only",
        "--skip-git-repo-check",
        "--ephemeral",
        "-C",
        str(repo_root),
        "--output-schema",
        str(schema_path),
        "--image",
        str(baseline_image),
        "--image",
        str(variant_image),
        "-o",
        str(last_message),
    ]
    if model:
        command.extend(["--model", model])
    command.append("-")

    prompt = _pair_prompt(row)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        return {
            **pair_base,
            "status": "failed",
            "error": f"visual review timed out after {timeout_seconds:.1f}s",
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    duration = round(time.monotonic() - started, 6)
    parsed, parse_error = _parse_visual_json(last_message, completed.stdout)
    if completed.returncode != 0 or parse_error:
        return {
            **pair_base,
            "status": "failed",
            "error": parse_error or f"codex exited with code {completed.returncode}",
            "command": command,
            "exit_code": completed.returncode,
            "duration_seconds": duration,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "last_message": str(last_message),
        }

    label = str(parsed.get("quality_label", "inconclusive"))
    if label not in QUALITY_LABELS:
        label = "inconclusive"
    return {
        **pair_base,
        "status": "completed",
        "qualitative_pass": bool(parsed.get("qualitative_pass", False)),
        "quality_label": label,
        "baseline_summary": str(parsed.get("baseline_summary", "")),
        "variant_summary": str(parsed.get("variant_summary", "")),
        "artifact_summary": str(parsed.get("artifact_summary", "")),
        "prompt_alignment": str(parsed.get("prompt_alignment", "")),
        "visible_regressions": _string_list(parsed.get("visible_regressions", [])),
        "confidence": str(parsed.get("confidence", "medium")),
        "command": command,
        "exit_code": completed.returncode,
        "duration_seconds": duration,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "last_message": str(last_message),
    }


def _pair_prompt(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You are the Checker Step visual evaluator.",
            "Two images are attached in this exact order:",
            "1. baseline output",
            "2. candidate optimized output",
            "",
            "Compare the candidate against the baseline and the prompt.",
            "Judge visible quality, prompt alignment, obvious artifacts, missing objects, color/layout drift, and semantic degradation.",
            "Do not run shell commands or edit files. Only inspect the attached images.",
            "Return only JSON that matches the provided schema.",
            "",
            f"Prompt: {row.get('prompt', '')}",
            f"Variant: {row.get('variant', '')}",
            f"PSNR: {row.get('psnr')}",
            f"MSE: {row.get('mse')}",
            f"MAE: {row.get('mae')}",
            f"DiT speedup: {row.get('dit_speedup')}",
            "",
            "Use `qualitative_pass=false` for major semantic loss, severe artifacts, unreadable subject, or poor prompt alignment.",
            "Use `quality_label=minor_regression` for visible but acceptable drift.",
            "Use `quality_label=similar` when the optimized output is broadly comparable to baseline.",
        ]
    )


def _output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "qualitative_pass": {"type": "boolean"},
            "quality_label": {
                "type": "string",
                "enum": sorted(QUALITY_LABELS),
            },
            "baseline_summary": {"type": "string"},
            "variant_summary": {"type": "string"},
            "artifact_summary": {"type": "string"},
            "prompt_alignment": {"type": "string"},
            "visible_regressions": {
                "type": "array",
                "items": {"type": "string"},
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
            },
        },
        "required": [
            "qualitative_pass",
            "quality_label",
            "baseline_summary",
            "variant_summary",
            "artifact_summary",
            "prompt_alignment",
            "visible_regressions",
            "confidence",
        ],
    }


def _parse_visual_json(last_message: Path, stdout: str) -> tuple[dict[str, Any], str]:
    candidates = []
    if last_message.exists():
        candidates.append(last_message.read_text(encoding="utf-8", errors="replace"))
    candidates.append(stdout)
    for text in candidates:
        parsed = _first_json_object(text)
        if isinstance(parsed, dict):
            return parsed, ""
    return {}, "Could not parse visual evaluator JSON output."


def _first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _overall_label(labels: set[str]) -> str:
    if "major_regression" in labels:
        return "major_regression"
    if "minor_regression" in labels:
        return "minor_regression"
    if "better" in labels:
        return "better"
    if "similar" in labels:
        return "similar"
    return "inconclusive"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
