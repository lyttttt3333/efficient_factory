from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from string import Template
from typing import Any


CANDIDATE_PATH = Path("draft_canvas/flux_schnell_candidate_canvas.py")


def main() -> int:
    workdir = Path(os.environ["EAL_WORKDIR"]).resolve()
    result_path = Path(os.environ["EAL_EXECUTOR_RESULT"]).resolve()
    action_path = Path(os.environ["EAL_REVIEWER_NEXT_ACTION"]).resolve()
    iteration = int(os.environ.get("EAL_ITERATION", "1"))
    started = time.monotonic()

    commands: list[dict] = []
    action = _read_json(action_path)
    selected_group, selected_variant = _selected_skill(action)

    commands.append(_run(
        "conda run -n auto_deploy_flux_eff python -B agent_loop/git_state_snapshot.py snapshot --output agent_loop_state/pre_executor_git_state.json",
        workdir,
    ))

    changed_files: list[str] = []
    changed_skill_ids: list[str] = []
    no_diff_reason = None
    status = "ready_for_checker"
    known_risks: list[str] = []

    parameter_settings: dict[str, Any] = {}

    if selected_group == "sparse_attention" and selected_variant in {"pisa", "spargeattn"}:
        candidate = _sparse_candidate_plan(action, selected_variant, iteration)
        before = _safe_read(workdir / CANDIDATE_PATH)
        content = _sparse_candidate_content(candidate)
        if before != content:
            (workdir / CANDIDATE_PATH).write_text(content, encoding="utf-8")
            changed_files.append(str(CANDIDATE_PATH))
            changed_skill_ids.append(selected_variant)
        else:
            no_diff_reason = (
                f"{selected_variant} candidate wrapper was already installed with the requested "
                f"parameters: {_compact_parameters(candidate)}."
            )
            changed_skill_ids.append(selected_variant)
        parameter_settings = {
            "sparse_attention": {
                "variant": selected_variant,
                "method": selected_variant,
                "apply_to": candidate["apply_to"],
                "min_tokens": candidate["min_tokens"],
                "max_tokens": candidate["max_tokens"],
                "density": candidate["density"],
                "block_size": candidate["block_size"],
                "topk": candidate["topk"],
                "parameter_search_move": candidate["parameter_search_move"],
            },
            selected_variant: {
                "method": selected_variant,
                "apply_to": candidate["apply_to"],
                "min_tokens": candidate["min_tokens"],
                "max_tokens": candidate["max_tokens"],
                "density": candidate["density"],
                "block_size": candidate["block_size"],
                "topk": candidate["topk"],
            },
        }
    else:
        status = "blocked"
        no_diff_reason = (
            "Deterministic executor adapter currently supports sparse_attention:pisa "
            "and sparse_attention:spargeattn; "
            f"reviewer requested {selected_group}:{selected_variant}."
        )
        known_risks.append(no_diff_reason)

    commands.append(_run(
        "conda run -n auto_deploy_flux_eff python -B benchmark/flux_schnell_benchmark.py --help",
        workdir,
    ))
    commands.append(_run(
        "conda run -n auto_deploy_flux_eff python -B -m py_compile "
        "draft_canvas/flux_schnell_candidate_canvas.py "
        "draft_canvas/flux_schnell_sparse_attention_canvas.py "
        "comfy_extras/nodes_sparse_attention.py "
        "benchmark/flux_schnell_benchmark.py",
        workdir,
    ))
    commands.append(_run(
        "conda run -n auto_deploy_flux_eff python -B agent_loop/git_state_snapshot.py diff "
        "--baseline agent_loop_state/pre_executor_git_state.json "
        "--output agent_loop_state/executor_diff.json",
        workdir,
    ))

    if any(command["exit_code"] != 0 for command in commands[1:]):
        status = "failed"
        known_risks.append("One or more executor smoke checks failed; see commands_run.")

    checker_commands = action.get("checker_commands")
    if not isinstance(checker_commands, list):
        checker_commands = []

    _write_json(
        result_path,
        {
            "schema_version": "auto_deploy.executor_result.v1",
            "role": "executor",
            "iteration_id": f"iter_{iteration:03d}",
            "iteration": iteration,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "reviewer_plan_path": str(action_path),
            "pre_executor_git_state_path": "agent_loop_state/pre_executor_git_state.json",
            "executor_diff_path": "agent_loop_state/executor_diff.json",
            "changed_files": changed_files,
            "changed_skill_ids": changed_skill_ids,
            "benchmark_interface_changes": [],
            "commands_run": commands,
            "smoke_results": [
                {
                    "name": "benchmark_help",
                    "passed": commands[1]["exit_code"] == 0,
                    "command": commands[1]["command"],
                },
                {
                    "name": "py_compile",
                    "passed": commands[2]["exit_code"] == 0,
                    "command": commands[2]["command"],
                },
            ],
            "checker_commands": checker_commands,
            "known_risks": known_risks,
            "no_diff_reason": no_diff_reason,
            "status": status,
            "summary": (
                f"Installed sparse_attention:{selected_variant} in the canonical candidate wrapper."
                if changed_files
                else no_diff_reason or "Executor completed without source changes."
            ),
            "artifacts": {
                "executor_adapter": "run_executor_from_action.py",
                "restricted_edits": [],
                "candidate_metadata": "draft_canvas/flux_schnell_candidate_canvas.py:CANDIDATE_METADATA",
                "parameter_settings": parameter_settings,
            },
            "executor_wrapper": {
                "mode": "deterministic_adapter",
                "duration_seconds": round(time.monotonic() - started, 6),
            },
        },
    )
    return 0 if status in {"ready_for_checker", "blocked"} else 1


def _run(command: str, cwd: Path) -> dict:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "duration_seconds": round(time.monotonic() - started, 6),
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _selected_skill(action: dict) -> tuple[str, str]:
    metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
    candidates = [
        {
            "group": action.get("selected_group"),
            "variant": action.get("selected_variant"),
        },
        metadata,
    ]
    for key in ("optimization_strategy", "model_context"):
        value = action.get(key)
        if not isinstance(value, dict):
            continue
        if key == "model_context":
            value = value.get("optimization_strategy", {})
            if not isinstance(value, dict):
                continue
        primary = value.get("primary_skill") or value.get("primary_candidate")
        if isinstance(primary, dict):
            candidates.append(primary)

    for candidate in candidates:
        group = str(candidate.get("group") or "").strip()
        variant = str(candidate.get("variant") or "").strip()
        if group and variant:
            return group, variant
    return "", ""


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _sparse_candidate_plan(action: dict, variant: str, iteration: int) -> dict[str, Any]:
    text = _previous_decision_text()
    if not text:
        text = str(action.get("instructions", "")).lower()
    visual_repair = any(
        token in text
        for token in (
            "repair visual",
            "visual regression",
            "visual quality regression",
            "qualitative review failed",
            "major_regression",
            "minor_regression",
            "psnr regression",
        )
    )
    speed_repair = any(token in text for token in ("speed below", "speedup", "more aggressive"))
    previous_attempts = _previous_variant_attempt_count(variant)

    if variant == "spargeattn":
        topk = 0.25
        if visual_repair:
            visual_schedule = [0.50, 0.75, 1.00]
            topk = visual_schedule[min(previous_attempts, len(visual_schedule) - 1)]
        elif speed_repair and iteration > 1:
            speed_schedule = [0.75, 0.50, 0.35, 0.25]
            speed_attempt = max(previous_attempts - 3, 0)
            topk = speed_schedule[min(speed_attempt, len(speed_schedule) - 1)]
        return {
            "skill_id": "spargeattn",
            "method": "spargeattn",
            "apply_to": "single",
            "min_tokens": 128,
            "max_tokens": 1_000_000,
            "density": 0.15,
            "block_size": 128,
            "topk": topk,
            "parameter_search_move": (
                "spargeattn visual-repair setting with higher topk"
                if visual_repair
                else "initial spargeattn official sparse-attention setting"
            ),
        }

    density_schedule = [0.15, 0.35, 0.60, 0.85, 1.00]
    density = density_schedule[min(previous_attempts, len(density_schedule) - 1)]
    if not visual_repair and speed_repair:
        density = 0.15
    return {
        "skill_id": "pisa",
        "method": "pisa",
        "apply_to": "single",
        "min_tokens": 128,
        "max_tokens": 1_000_000,
        "density": density,
        "block_size": 128,
        "topk": 0.25,
        "parameter_search_move": (
            "increase PISA density after visual/quality regression to test a less destructive point"
            if visual_repair and iteration > 1
            else "initial official-resolution PISA setting using the skill-card density"
        ),
    }


def _previous_decision_text() -> str:
    parts = [
        os.environ.get("EAL_PREVIOUS_DECISION", ""),
        os.environ.get("EAL_PREVIOUS_INSTRUCTIONS", ""),
    ]
    previous_path = Path(os.environ.get("EAL_PREVIOUS_REVIEWER_DECISION", ""))
    if previous_path.exists():
        try:
            previous = _read_json(previous_path)
        except (OSError, json.JSONDecodeError):
            previous = {}
        parts.extend(
            str(value)
            for value in (
                previous.get("rationale", ""),
                previous.get("next_instructions", ""),
            )
        )
        hint = previous.get("next_action_hint", {})
        if isinstance(hint, dict):
            parts.append(str(hint.get("reason", "")))
    return " ".join(part for part in parts if part).lower()


def _previous_variant_attempt_count(variant: str) -> int:
    run_dir = Path(os.environ.get("EAL_RUN_DIR", ""))
    run_root = run_dir.resolve().parent if str(run_dir) else Path()
    if not run_root.exists():
        return 0
    count = 0
    for path in sorted(run_root.glob("iter-*/executor_result.json")):
        try:
            data = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        artifacts = data.get("artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        settings = artifacts.get("parameter_settings", {})
        if not isinstance(settings, dict):
            continue
        sparse = settings.get("sparse_attention", {})
        if isinstance(sparse, dict) and sparse.get("variant") == variant:
            count += 1
    return count


def _compact_parameters(candidate: dict[str, Any]) -> str:
    keys = ("method", "apply_to", "density", "block_size", "topk", "min_tokens", "max_tokens")
    return ", ".join(f"{key}={candidate[key]}" for key in keys if key in candidate)


def _sparse_candidate_content(candidate: dict[str, Any]) -> str:
    return Template('''from __future__ import annotations

import math

from draft_canvas.flux_schnell_cache_canvas import build_flux_schnell_baseline
from efficient_skill.common.workflow import Workflow, clone_workflow, output_ref
from efficient_skill.sparse_attention import insert_sparse_attention
from model.flux_schnell import FLUX_SCHNELL_SPLIT


CANDIDATE_NAME = "executor_candidate"

SPARSE_SKILL_ID = "$skill_id"
SPARSE_METHOD = "$method"
SPARSE_APPLY_TO = "$apply_to"
SPARSE_MIN_TOKENS = $min_tokens
SPARSE_MAX_TOKENS = $max_tokens
SPARSE_DENSITY = $density
SPARSE_BLOCK_SIZE = $block_size
SPARSE_TOPK = $topk
SPARSE_DENSE_INITIAL_STEPS = 0
SPARSE_PRECOMPILE_HEADS = 24
SPARSE_PRECOMPILE_HEAD_DIM = 128
SPARSE_TEXT_TOKENS = 256
PARAMETER_SEARCH_MOVE = $parameter_search_move

OFFICIAL_FLUX_SCHNELL_CONFIG = {
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "guidance_scale": 0.0,
    "max_sequence_length": 256,
    "sampler_name": "euler",
    "scheduler": "simple",
    "time_shift": "not_exposed_by_basic_scheduler",
}


def _flux_attention_tokens(width: int, height: int, text_tokens: int = SPARSE_TEXT_TOKENS) -> int:
    latent_w = max(1, math.ceil(width / 16))
    latent_h = max(1, math.ceil(height / 16))
    return latent_w * latent_h + text_tokens


def _pisa_parameter_settings(
    width: int = OFFICIAL_FLUX_SCHNELL_CONFIG["width"],
    height: int = OFFICIAL_FLUX_SCHNELL_CONFIG["height"],
    text_tokens: int = SPARSE_TEXT_TOKENS,
) -> dict[str, object]:
    return {
        "skill_id": SPARSE_SKILL_ID,
        "method": SPARSE_METHOD,
        "apply_to": SPARSE_APPLY_TO,
        "min_tokens": SPARSE_MIN_TOKENS,
        "max_tokens": SPARSE_MAX_TOKENS,
        "density": SPARSE_DENSITY,
        "block_size": SPARSE_BLOCK_SIZE,
        "topk": SPARSE_TOPK,
        "dense_initial_steps": SPARSE_DENSE_INITIAL_STEPS,
        "precompile_tokens": _flux_attention_tokens(width, height, text_tokens),
        "precompile_tokens_formula": "ceil(width / 16) * ceil(height / 16) + max_sequence_length",
        "max_sequence_length_assumed_for_precompile": text_tokens,
        "precompile_heads": SPARSE_PRECOMPILE_HEADS,
        "precompile_head_dim": SPARSE_PRECOMPILE_HEAD_DIM,
    }


CANDIDATE_METADATA = {
    "candidate_name": CANDIDATE_NAME,
    "stack": [f"sparse_attention:{SPARSE_SKILL_ID}"],
    "stack_description": f"Baseline Flux Schnell split workflow plus {SPARSE_METHOD} sparse attention on single-stream Flux DiT attention blocks.",
    "changed_skill_ids": [SPARSE_SKILL_ID],
    "parameter_search_move": PARAMETER_SEARCH_MOVE,
    "official_benchmark_config": OFFICIAL_FLUX_SCHNELL_CONFIG,
    "parameter_settings": {
        SPARSE_SKILL_ID: _pisa_parameter_settings(),
    },
    "composition_status": {
        "attention": {
            "included": True,
            "skill_id": SPARSE_SKILL_ID,
            "reason": "primary reviewer candidate for this iteration",
        },
        "ffn_linear_quantization": {
            "included": False,
            "status": "not_applied_this_iteration",
            "next_stack_role": "selective TorchAO FFN/Linear quantization remains available for later composition",
        },
        "compile": {
            "included": False,
            "status": "not_applied_this_iteration",
            "next_stack_role": "torch_compile remains available for later composition after PISA validity is measured",
        },
        "cache_reuse": {
            "included": False,
            "status": "not_applied_this_iteration",
            "next_stack_role": "reuse-cache skills remain available only if visual quality holds under fixed-baseline benchmark",
        },
    },
    "checker_interface": "benchmark/flux_schnell_benchmark.py evaluates this wrapper without skill-specific flags",
}


def prepare_candidate(
    *,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    max_sequence_length: int,
    sampler_name: str,
    scheduler: str,
) -> dict:
    return {
        "prepared": True,
        "prepared_scope": "metadata_and_pisa_precompile_parameters_only",
        "candidate_name": CANDIDATE_NAME,
        "stack": list(CANDIDATE_METADATA["stack"]),
        "parameter_settings": {
            SPARSE_SKILL_ID: _pisa_parameter_settings(width, height, max_sequence_length),
        },
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "max_sequence_length": max_sequence_length,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
    }


def clear_candidate_state() -> None:
    return None


def build_flux_schnell_candidate(
    prompt: str,
    seed: int,
    diffusion_name: str = FLUX_SCHNELL_SPLIT.diffusion.filename,
    t5xxl_name: str = FLUX_SCHNELL_SPLIT.t5xxl.filename,
    clip_l_name: str = FLUX_SCHNELL_SPLIT.clip_l.filename,
    vae_name: str = FLUX_SCHNELL_SPLIT.vae.filename,
    diffusion_weight_dtype: str = "default",
    clip_device: str = "default",
    width: int = 1024,
    height: int = 1024,
    steps: int = 4,
    guidance: float = 0.0,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    filename_prefix: str = "flux_executor_candidate",
) -> Workflow:
    workflow = clone_workflow(
        build_flux_schnell_baseline(
            prompt=prompt,
            seed=seed,
            diffusion_name=diffusion_name,
            t5xxl_name=t5xxl_name,
            clip_l_name=clip_l_name,
            vae_name=vae_name,
            diffusion_weight_dtype=diffusion_weight_dtype,
            clip_device=clip_device,
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            sampler_name=sampler_name,
            scheduler=scheduler,
            filename_prefix=filename_prefix,
        )
    )
    sparse_settings = _pisa_parameter_settings(width, height, SPARSE_TEXT_TOKENS)
    insert_sparse_attention(
        workflow,
        model_ref=output_ref("1", 0),
        method=SPARSE_METHOD,
        apply_to=SPARSE_APPLY_TO,
        min_tokens=SPARSE_MIN_TOKENS,
        max_tokens=SPARSE_MAX_TOKENS,
        density=SPARSE_DENSITY,
        block_size=SPARSE_BLOCK_SIZE,
        topk=SPARSE_TOPK,
        precompile_tokens=int(sparse_settings["precompile_tokens"]),
        precompile_heads=SPARSE_PRECOMPILE_HEADS,
        precompile_head_dim=SPARSE_PRECOMPILE_HEAD_DIM,
        verbose=False,
    )
    return workflow
''').substitute(
        skill_id=candidate["skill_id"],
        method=candidate["method"],
        apply_to=candidate["apply_to"],
        min_tokens=candidate["min_tokens"],
        max_tokens=candidate["max_tokens"],
        density=repr(float(candidate["density"])),
        block_size=candidate["block_size"],
        topk=repr(float(candidate["topk"])),
        parameter_search_move=json.dumps(candidate["parameter_search_move"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
