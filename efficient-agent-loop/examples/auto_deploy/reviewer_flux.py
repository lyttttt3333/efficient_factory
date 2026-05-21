from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_SKILLS = {
    "cache": ["periodic_reuse", "similarity_reuse", "delta_reuse", "ema_reuse"],
    "quantization": ["selective_torchao_nvfp4", "selective_torchao_fp8"],
    "sparse_attention": ["pisa", "spargeattn"],
    "compile": ["torch_compile"],
}

UNIFIED_FLUX_BENCHMARK = "benchmark/flux_schnell_benchmark.py"
CANDIDATE_CANVAS = "draft_canvas/flux_schnell_candidate_canvas.py"

EDIT_AREAS = {
    "cache": [
        "efficient_skill/cache/",
        CANDIDATE_CANVAS,
        "draft_canvas/flux_schnell_cache_canvas.py",
        "comfy_extras/nodes_reuse_cache.py",
        UNIFIED_FLUX_BENCHMARK,
    ],
    "quantization": [
        "efficient_skill/quantization/",
        CANDIDATE_CANVAS,
        "draft_canvas/flux_schnell_quantization_canvas.py",
        "comfy_extras/nodes_quantization.py",
        UNIFIED_FLUX_BENCHMARK,
    ],
    "sparse_attention": [
        "efficient_skill/sparse_attention/",
        CANDIDATE_CANVAS,
        "draft_canvas/flux_schnell_sparse_attention_canvas.py",
        "comfy_extras/nodes_sparse_attention.py",
        UNIFIED_FLUX_BENCHMARK,
    ],
}

RESTRICTED_EDIT_PREFIXES = {
    "efficient_skill": "efficient_skill/",
    "benchmark": "benchmark/",
}

ALLOWED_RESTRICTED_REASONS = [
    "missing_skill_capability_or_interface",
    "reusable_skill_behavior",
    "benchmark_validity",
]

BASELINE_ARTIFACT_NAME = "eal_flux_schnell_official_baseline"

OFFICIAL_FLUX_BENCHMARK_CONFIG = {
    "model_family": "FLUX.1 schnell",
    "source": "FLUX.1-schnell model card plus Diffusers FluxPipeline defaults unless user override",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "guidance": 0.0,
    "guidance_scale": 0.0,
    "cfg": "not_applicable_for_schnell_unless_workflow_exposes_true_cfg",
    "max_sequence_length": 256,
    "sampler": "official_model_or_workflow_default",
    "scheduler": "official_model_or_workflow_default",
    "time_shift": "official_scheduler_or_workflow_default_must_be_recorded",
    "override_rule": (
        "Only an explicit user request or a repo/model-card official default may "
        "override this benchmark config; legacy 512x512 smoke settings are not "
        "acceptance evidence."
    ),
}


def main() -> None:
    context = os.environ.get("EAL_REVIEWER_CONTEXT", "before_executor")
    if context == "after_checker":
        _write_decision()
        return
    _write_next_action()


def _write_next_action() -> None:
    workdir = Path(os.environ["EAL_WORKDIR"]).resolve()
    action_path = Path(os.environ["EAL_REVIEWER_NEXT_ACTION"])
    experiment = _read_json(Path(os.environ["EAL_EXPERIMENT_JSON"]))
    iteration = int(os.environ.get("EAL_ITERATION", "1"))

    text = " ".join(
        str(experiment.get(key, "")) for key in ("name", "goal", "instructions")
    ).lower()
    metadata = experiment.get("metadata") if isinstance(experiment.get("metadata"), dict) else {}

    readme_status = {
        "reviewer": (workdir / "README_REVIEWER.md").exists(),
        "executor": (workdir / "README_EXECUTOR.md").exists(),
        "checker": (workdir / "README_CHECKER.md").exists(),
    }
    skill_index = workdir / "efficient_skill" / "SKILL_INDEX.md"
    skills = _parse_skill_index(skill_index) if skill_index.exists() else DEFAULT_SKILLS
    hint = _previous_action_hint()
    group = str(hint.get("group") or "").strip()
    if group not in skills:
        group = _choose_group(str(metadata.get("group", "")).lower(), text, skills)
    variant = str(hint.get("variant") or "").strip()
    if variant not in (skills.get(group) or []):
        variant = _choose_variant(str(metadata.get("variant", "")).strip(), group, skills)
    repair_reason = str(hint.get("reason", "")).strip()
    previous_guidance = str(hint.get("next_instructions", "")).strip()
    previous_rationale = str(hint.get("rationale", "")).strip()
    previous = _find_previous_metrics(workdir, variant)
    parameter_blockers = _invalid_parameter_blockers(workdir, group, variant)
    override = _before_executor_candidate_override(
        workdir=workdir,
        group=group,
        variant=variant,
        skills=skills,
        previous_metric=previous,
        repair_reason=repair_reason,
        parameter_blockers=parameter_blockers,
    )
    if override:
        group = override["group"]
        variant = override["variant"]
        repair_reason = override["reason"]
        previous = _find_previous_metrics(workdir, variant)
        parameter_blockers = _invalid_parameter_blockers(workdir, group, variant)
    artifact_name = f"eal_checker_candidate_iter{iteration:03d}"
    benchmark = UNIFIED_FLUX_BENCHMARK
    previous = _find_previous_metrics(workdir, variant)
    diff_stat = _git(["diff", "--stat"], workdir)
    previous_decision = os.environ.get("EAL_PREVIOUS_DECISION", "").upper()
    target_role = (
        "Checker Step"
        if previous_decision == "NEEDS_RETEST" and _is_checker_retest_reason(repair_reason)
        else "Executor Agent"
    )

    checker_command = _checker_command(group, benchmark, variant, artifact_name)
    files = _combined_edit_areas(group)
    default_edit_files = _default_edit_areas(group)
    restricted_edit_files = _restricted_edit_areas(group)
    baseline_policy = _baseline_policy()
    official_config_instruction = _official_config_instruction()
    source_edit_policy = _source_edit_policy(
        repair_reason,
        authorize_benchmark_for_baseline=False,
    )
    parameter_search_policy = _parameter_search_policy(group, variant)
    final_delivery_policy = _final_delivery_policy(skills)
    authorized_edit_files = _authorized_edit_areas(
        default_edit_files,
        restricted_edit_files,
        source_edit_policy,
    )
    optimization_strategy = {
        "objective": "maximize aggregate Flux speedup under correctness and multimodal visual quality constraints",
        "primary_skill": {"group": group, "variant": variant},
        "available_skill_groups": skills,
        "combination_policy": [
            "Treat cache/reuse, quantization, sparse attention, compilation, and workflow-level changes as composable candidates.",
            "Do not stop at the first single skill that produces a local speedup.",
            "Treat every skill as a parameterized family, not a single yes/no candidate.",
            "Use directed parameter search: if speedup is small and quality is strong, make the next setting more aggressive; if quality regresses too much, back off toward the last passing setting.",
            "Prefer changes that remain compatible with later skill combinations.",
            "The final deliverable is a composed Flux candidate stack, not a report that individual skills were explored.",
            "If a single skill is fast but visually degraded, fix quality or combine a less destructive complementary skill instead of accepting.",
            "If a skill produces abnormal or internally inconsistent evidence, keep the same candidate active, ask Executor to diagnose/fix it, then retest before marking that skill exhausted or switching away.",
            "After abnormal evidence is resolved, continue parameter and composition search rather than accepting or abandoning a skill from one problematic run.",
            "Accept only when a final composed stack passes speed, correctness, required multimodal qualitative review, parameter-search frontier checks, and no obvious compatible next stack improvement remains.",
        ],
        "parameter_search_policy": parameter_search_policy,
        "final_delivery_policy": final_delivery_policy,
    }
    if target_role == "Checker Step":
        instructions = "\n".join(
            [
                "Rerun Checker for the same Flux candidate; do not modify code.",
                f"Candidate: `{group}` variant `{variant}`.",
                *( [f"Previous Reviewer hint: {repair_reason}."] if repair_reason else [] ),
                *( [f"Previous Reviewer instructions: {previous_guidance}"] if previous_guidance else [] ),
                *( [f"Previous Reviewer rationale: {previous_rationale}"] if previous_rationale else [] ),
                "Read `EAL_AGENT_INITIAL_PROMPT`, `EAL_AGENT_README`, and `EAL_REVIEWER_NEXT_ACTION` first.",
                "Keep prompts, seeds, official model hyperparameters, image metrics, fixed baseline, and DiT-only timing unchanged.",
                official_config_instruction,
                "If metrics show legacy smoke resolution/steps without a user override, report benchmark_valid=false; otherwise record missing secondary hyperparameter metadata for Reviewer follow-up.",
                "The benchmark result must include metrics.json, stdout/stderr/exit code, and the required multimodal qualitative judgment.",
                "If the benchmark command fails again or metrics.json is missing, report benchmark_valid=false and preserve the failure reason for Reviewer.",
                "",
                "Checker should run:",
                checker_command,
            ]
        )
    else:
        instructions = "\n".join(
            [
                "Implement the next controlled Flux optimization step toward the best composable skill stack.",
                f"Primary candidate for this iteration: `{group}` variant `{variant}`.",
                *( [f"Previous Reviewer hint: {repair_reason}."] if repair_reason else [] ),
                *( [f"Previous Reviewer instructions: {previous_guidance}"] if previous_guidance else [] ),
                *( [f"Previous Reviewer rationale: {previous_rationale}"] if previous_rationale else [] ),
                *_parameter_blocker_instruction_lines(parameter_blockers),
                "Read `EAL_AGENT_INITIAL_PROMPT`, `EAL_AGENT_README`, and `EAL_REVIEWER_NEXT_ACTION` first.",
                "Keep Flux benchmark assumptions fixed: prompts, seeds, image metrics, official model hyperparameters, and DiT-only timing.",
                official_config_instruction,
                "If the benchmark or canvas cannot expose secondary official hyperparameters such as guidance/CFG/sampler/scheduler/time_shift, prefer recording what is available and note the gap instead of blocking the whole run.",
                "Consider combinations across cache/reuse, quantization, sparse attention, compilation, and workflow-level skills.",
                "Do not stop at one isolated skill just because it improves a local metric; keep the implementation composable with complementary skills.",
                "Parameter search is required. Do not treat a skill id as exhausted after one setting; adjust parameters based on speed/quality evidence.",
                "Use the parameter search policy: weak speedup with good quality means try a more aggressive setting; strong speedup with visual/PSNR regression means back off to an intermediate setting.",
                "Record the exact parameters used for this candidate in benchmark metrics and in `executor_result.json` artifacts when you change implementation wiring.",
                "Final delivery target is one composed implementation stack, for example tuned sparse attention + torch compile + selective FFN/Linear quantization, with any incompatible component explicitly ruled out by evidence.",
                "Do not finish only because each skill family has been sampled once; finish only after the composed delivery stack is implemented and benchmarked.",
                "If the previous candidate was fast but visually degraded, reduce the destructive part of that skill or combine with a quality-preserving alternative before retesting.",
                "If the previous candidate produced abnormal or internally inconsistent evidence, diagnose and repair that same candidate before switching skills; after it is valid, continue searching its parameters and compatible combinations.",
                "Use a run-level fixed baseline: benchmark should create or reuse one baseline artifact for the fixed prompt/seed/official-hyperparameter/timing scope, then compare each candidate against that baseline.",
                "Do not remeasure baseline for every candidate iteration unless the baseline artifact is missing, corrupt, or the fixed official benchmark configuration changed.",
                f"Baseline artifact target: benchmark/artifacts/{BASELINE_ARTIFACT_NAME}.",
                f"Put the runnable optimized model behind `{CANDIDATE_CANVAS}`. The checker evaluates only that candidate wrapper and must not receive skill-specific flags.",
                f"Use the unified Flux benchmark interface only: `{UNIFIED_FLUX_BENCHMARK}`. Do not create or route formal checker runs through skill-family benchmark scripts.",
                f"Inspect the relevant files for the selected combination: {', '.join(files)}.",
                f"Default editable files this iteration: {', '.join(default_edit_files)}.",
                "Restricted edit surfaces: `efficient_skill/` reusable skill source and `benchmark/` checker code.",
                "Do not modify a restricted surface unless `source_edit_policy.current_authorization` explicitly authorizes that surface.",
                "Allowed restricted edit reasons are: missing_skill_capability_or_interface, reusable_skill_behavior, benchmark_validity.",
                _authorization_summary(source_edit_policy),
                "If a restricted edit seems necessary but is not authorized, leave it unchanged and explain the blocked change in `executor_result.json`.",
                "Prefer a new explicit variant or wrapper over changing an existing skill default.",
                "Write `executor_result.json` with changed files, smoke checks, and any no-diff reason.",
                "If you touch a restricted surface, record the surface, changed paths, and authorization reason under `artifacts.restricted_edits`.",
                "",
                "Checker should run:",
                checker_command,
            ]
        )

    action = {
        "iteration": iteration,
        "source": "auto_deploy_flux_reviewer",
        "target_role": target_role,
        "instructions": instructions,
        "previous_decision": os.environ.get("EAL_PREVIOUS_DECISION", ""),
        "reviewer_initial_prompt": os.environ.get("EAL_REVIEWER_INITIAL_PROMPT", ""),
        "reviewer_readme": os.environ.get("EAL_REVIEWER_README", ""),
        "target_initial_prompt": os.environ.get(
            "EAL_CHECKER_INITIAL_PROMPT"
            if target_role == "Checker Step"
            else "EAL_EXECUTOR_INITIAL_PROMPT",
            "",
        ),
        "target_readme": os.environ.get(
            "EAL_CHECKER_README"
            if target_role == "Checker Step"
            else "EAL_EXECUTOR_README",
            "",
        ),
        "model_context": {
            "repo": str(workdir),
            "model_family": "Flux Schnell split ComfyUI workflow",
            "skill_index": str(skill_index),
            "readmes_found": readme_status,
            "selected_group": group,
            "selected_variant": variant,
            "benchmark": benchmark,
            "artifact_name": artifact_name,
            "baseline_artifact_name": BASELINE_ARTIFACT_NAME,
            "candidate_interface": CANDIDATE_CANVAS,
            "official_model_config": _official_model_config(),
            "optimization_strategy": optimization_strategy,
            "parameter_search_policy": parameter_search_policy,
            "final_delivery_policy": final_delivery_policy,
        },
        "baseline_policy": baseline_policy,
        "source_edit_policy": source_edit_policy,
        "parameter_search_policy": parameter_search_policy,
        "final_delivery_policy": final_delivery_policy,
        "evidence": {
            "git_diff_stat": diff_stat,
            "available_skills": skills,
            "previous_metric": previous,
            "parameter_blockers": parameter_blockers,
        },
        "files_to_inspect_or_edit": files,
        "default_edit_files": default_edit_files,
        "restricted_edit_files": restricted_edit_files,
        "files_authorized_to_edit": authorized_edit_files,
        "optimization_strategy": optimization_strategy,
        "checker_commands": [checker_command],
        "visual_quality": {
            "enabled": True,
            "required": True,
            "evaluator": "codex",
            "max_pairs": 1,
            "timeout_seconds": 300,
        },
        "acceptance_criteria": [
            "Executor does not change PROMPTS, SEEDS, compare_images, or DiT timing semantics.",
            "Executor keeps changes scoped to the selected skill combination and records all changed_skill_ids.",
            "Executor treats `efficient_skill/` and `benchmark/` as restricted edit surfaces unless explicitly authorized by source_edit_policy.",
            "Executor records any restricted edits and authorization reasons under artifacts.restricted_edits.",
            "Executor treats skills as a composable optimization stack, not a one-off single-skill pass.",
            "Executor records concrete parameter settings for each candidate and keeps them visible in metrics or artifacts.",
            "Reviewer should not mark a skill exhausted after one parameter setting; it should direct coarse-to-fine parameter search based on speed/quality evidence.",
            "Reviewer should keep the same candidate active when Checker rejects abnormal or internally inconsistent evidence, route diagnosis/fix to Executor, then require Checker to validate speed and benchmark validity again before switching candidates.",
            "Reviewer must not ACCEPT until a final composed delivery stack has been implemented and benchmarked, or every incompatible stack component has a measured blocker.",
            "Final delivery must consider sparse attention, selective FFN/Linear quantization, compile, and compatible cache/reuse composition rather than ending after isolated skill exploration.",
            "Checker benchmark writes metrics.json and summary.md under benchmark/artifacts.",
            f"Checker formal acceptance runs use `{UNIFIED_FLUX_BENCHMARK}` to evaluate the single candidate wrapper in `{CANDIDATE_CANVAS}`.",
            "Checker commands must not include skill-specific names or parameters such as sparse attention, quantization recipe, density, topk, or apply_to.",
            "Skill-family benchmark scripts and skill-specific benchmark flags must not be used as Reviewer acceptance evidence.",
            "Checker benchmark uses a run-level fixed baseline artifact instead of remeasuring baseline for each candidate iteration.",
            "Checker benchmark config matches the primary official Flux Schnell setup: 1024x1024 and 4 steps; secondary fields such as guidance_scale, max_sequence_length, sampler/scheduler/time_shift should be recorded when available.",
            "Checker reports benchmark_valid=false if a valid fixed baseline exists but the candidate result is compared against a newly measured per-iteration baseline.",
            "Checker reports benchmark_valid=true only if git diff is unchanged during benchmark.",
            "Checker reports speed and quality from metrics.json, not from executor claims.",
            "Checker must attach baseline and variant images to a multimodal evaluator and write a qualitative visual judgment.",
            "Reviewer must not ACCEPT if qualitative visual judgment is missing, inconclusive, failed, or reports a major visible regression.",
            "Reviewer should prefer NEEDS_FIX over ACCEPT when an obvious compatible skill combination remains and the current result only tests one isolated skill.",
            "Reviewer should keep iterating until the candidate space is exhausted or no compatible next optimization remains.",
        ],
    }
    _write_json(action_path, action)


def _write_decision() -> None:
    checker_path = Path(os.environ["EAL_CHECKER_RESULT"])
    diff_path = Path(os.environ["EAL_GIT_DIFF"])
    decision_path = Path(os.environ["EAL_REVIEWER_DECISION"])
    executor_path_value = os.environ.get("EAL_EXECUTOR_RESULT", "")
    action_path_value = os.environ.get("EAL_REVIEWER_NEXT_ACTION", "")
    executor_path = Path(executor_path_value) if executor_path_value else None
    action_path = Path(action_path_value) if action_path_value else None
    workdir = Path(os.environ["EAL_WORKDIR"]).resolve()

    checker = _read_json(checker_path) if checker_path.exists() else {}
    executor = _read_json(executor_path) if executor_path and executor_path.is_file() else {}
    action = _read_json(action_path) if action_path and action_path.is_file() else {}
    diff = diff_path.read_text(encoding="utf-8") if diff_path.exists() else ""
    diff_present = bool(diff.strip())
    command = checker.get("command", {}) if isinstance(checker.get("command"), dict) else {}
    exit_code = command.get("exit_code", 1)
    executor_command = executor.get("command", {}) if isinstance(executor.get("command"), dict) else {}
    executor_exit = executor_command.get("exit_code", 0)
    no_diff_reason = str(executor.get("no_diff_reason", "")).strip()
    no_diff_reason_source = str(executor.get("no_diff_reason_source", "")).strip()
    no_diff_explained = bool(no_diff_reason) and no_diff_reason_source != "scheduler_fallback"

    recommendation = str(checker.get("recommendation", "")).upper()
    benchmark_valid = bool(checker.get("benchmark_valid", exit_code == 0))
    implementation_valid = bool(checker.get("implementation_valid", False))
    benchmark_exit_code = _benchmark_exit_code(checker, exit_code)
    benchmark_failure_reason = _benchmark_failure_reason(checker, benchmark_exit_code)
    qualitative = checker.get("qualitative", {})
    qualitative_overall = (
        qualitative.get("overall", {}) if isinstance(qualitative, dict) else {}
    )
    qualitative_status = str(qualitative.get("status", "")) if isinstance(qualitative, dict) else ""
    qualitative_required = bool(qualitative.get("required", False)) if isinstance(qualitative, dict) else False
    qualitative_pass = qualitative_overall.get("qualitative_pass")
    quality_label = str(qualitative_overall.get("quality_label", ""))
    selected = _selected_candidate(action)
    parameter_failure_reason = _parameter_resource_failure_reason(selected, checker, executor)
    if parameter_failure_reason:
        benchmark_failure_reason = parameter_failure_reason
    evidence = _decision_evidence(
        checker=checker,
        diff_present=diff_present,
        no_diff_reason=no_diff_reason,
        selected=selected,
    )
    rationale_prefix = _rationale_prefix(evidence)
    abnormal_reason = _abnormal_evidence_reason(checker)
    parameter_regression_reason = _parameter_regression_reason(
        workdir,
        action,
        checker,
        selected,
    )
    density_direction_blocker = _pisa_density_direction_blocker(
        workdir,
        selected,
        checker,
    )
    parameter_blocker = _parameter_frontier_blocker(workdir, selected, checker)

    if executor_exit != 0:
        decision = {
            "decision": "NEEDS_FIX",
            "rationale": f"Executor failed before Checker evidence can be trusted: exit_code={executor_exit}.",
            "next_instructions": "Fix the Executor failure and rerun the same candidate.",
            "next_action_hint": {**selected, "reason": "repair executor failure"},
        }
    elif not diff_present and not no_diff_explained:
        decision = {
            "decision": "NEEDS_FIX",
            "rationale": (
                "Executor produced no git diff and did not provide a specific no_diff_reason. "
                + rationale_prefix
            ).strip(),
            "next_instructions": (
                "Either implement the requested optimization change, or explicitly explain why "
                "this iteration is verification-only in executor_result.json."
            ),
            "next_action_hint": {**selected, "reason": "no source diff was not explained"},
        }
    elif not checker.get("git_diff_unchanged", True):
        decision = {
            "decision": "NEEDS_RETEST",
            "rationale": "Checker changed the git diff while benchmarking, so the benchmark is polluted.",
            "next_instructions": "Rerun Checker after removing benchmark side effects from the git diff.",
            "next_action_hint": {**selected, "reason": "benchmark pollution requires retest"},
        }
    elif quality_label == "major_regression" or qualitative_pass is False:
        decision = {
            "decision": "NEEDS_FIX",
            "rationale": (
                "Required multimodal qualitative review failed. "
                + rationale_prefix
            ).strip(),
            "next_instructions": _quality_fix_instructions(selected, checker),
            "next_action_hint": {
                **selected,
                "reason": "repair visual quality regression before exploring farther",
            },
        }
    elif recommendation == "NEEDS_RETEST" or not benchmark_valid:
        if benchmark_exit_code != 0:
            decision = {
                "decision": "NEEDS_FIX",
                "rationale": (
                    str(checker.get("rationale", "Checker reported a failed benchmark command."))
                    + " "
                    + rationale_prefix
                ).strip(),
                "next_instructions": _benchmark_failure_fix_instructions(
                    selected,
                    benchmark_failure_reason,
                ),
                "next_action_hint": {**selected, "reason": benchmark_failure_reason},
            }
        else:
            decision = {
                "decision": "NEEDS_RETEST",
                "rationale": (
                    str(checker.get("rationale", "Checker reported an invalid benchmark."))
                    + " "
                    + rationale_prefix
                ).strip(),
                "next_instructions": "Rerun an uncontaminated Flux benchmark with warmup and valid artifacts.",
                "next_action_hint": {**selected, "reason": benchmark_failure_reason},
            }
    elif qualitative_required and (
        not qualitative or qualitative_status != "completed" or qualitative_pass is not True
    ):
        decision = {
            "decision": "NEEDS_RETEST",
            "rationale": (
                "Required multimodal qualitative review is missing, failed to run, or inconclusive. "
                + rationale_prefix
            ).strip(),
            "next_instructions": "Rerun Checker with baseline/candidate image qualitative review enabled.",
            "next_action_hint": {**selected, "reason": "qualitative review must complete"},
        }
    elif parameter_regression_reason:
        decision = {
            "decision": "NEEDS_FIX",
            "rationale": (parameter_regression_reason + " " + rationale_prefix).strip(),
            "next_instructions": _parameter_regression_instructions(
                selected,
                checker,
                parameter_regression_reason,
            ),
            "next_action_hint": {
                **selected,
                "reason": parameter_regression_reason,
            },
        }
    elif density_direction_blocker and _speed_below_target_with_passing_quality(checker):
        next_hint = _pisa_density_blocked_next_hint(action, density_direction_blocker)
        decision = {
            "decision": "NEEDS_FIX",
            "rationale": (density_direction_blocker + " " + rationale_prefix).strip(),
            "next_instructions": _pisa_density_blocked_instructions(
                next_hint,
                checker,
                density_direction_blocker,
            ),
            "next_action_hint": next_hint,
        }
    elif parameter_blocker and _speed_below_target_with_passing_quality(checker):
        next_hint = _next_candidate_hint(workdir, action, checker, reason=parameter_blocker)
        decision = {
            "decision": "NEEDS_FIX",
            "rationale": (parameter_blocker + " " + rationale_prefix).strip(),
            "next_instructions": _exploration_instructions(next_hint)
            if next_hint
            else (
                f"Record `{selected.get('group')}:{selected.get('variant')}` as blocked by measured "
                "speed/quality frontier evidence, then build the best remaining composed stack or explain "
                "why no compatible candidate remains."
            ),
            "next_action_hint": next_hint
            or {**selected, "reason": "parameter frontier blocked and no next compatible candidate found"},
        }
    elif _speed_below_target_with_passing_quality(checker):
        decision = {
            "decision": "NEEDS_FIX",
            "rationale": (
                "Benchmark structure and visual quality are valid, but candidate speed is below the target. "
                + rationale_prefix
            ).strip(),
            "next_instructions": _speed_parameter_search_instructions(selected, checker),
            "next_action_hint": {
                **selected,
                "reason": "quality passed but speed below target; continue directed parameter search",
            },
        }
    elif recommendation == "REJECT" and abnormal_reason:
        decision = {
            "decision": "NEEDS_FIX",
            "rationale": (
                str(checker.get("rationale", "Checker rejected abnormal implementation evidence."))
                + " "
                + rationale_prefix
            ).strip(),
            "next_instructions": _abnormal_evidence_fix_instructions(
                selected,
                abnormal_reason,
            ),
            "next_action_hint": {**selected, "reason": abnormal_reason},
        }
    elif recommendation == "REJECT":
        decision = {
            "decision": "NEEDS_FIX",
            "rationale": (
                str(checker.get("rationale", "Checker rejected the implementation evidence."))
                + " "
                + rationale_prefix
            ).strip(),
            "next_instructions": "Treat the rejected implementation as a failed candidate and try the next compatible skill or a less destructive variant.",
            "next_action_hint": _next_candidate_hint(workdir, action, checker, reason="current candidate was rejected"),
        }
    elif recommendation == "STOP":
        decision = {
            "decision": "STOP",
            "rationale": checker.get("rationale", "Checker requested STOP."),
        }
    elif recommendation == "ACCEPT" and implementation_valid:
        next_hint = _next_candidate_hint(
            workdir,
            action,
            checker,
            reason="current candidate passed; continue exploring compatible remaining skills",
        )
        if next_hint:
            decision = {
                "decision": "NEEDS_FIX",
                "rationale": (
                    "Current candidate passed speed, quantitative quality, and qualitative review, "
                    "but the compatible optimization space is not exhausted. "
                    + rationale_prefix
                ).strip(),
                "next_instructions": _exploration_instructions(next_hint),
                "next_action_hint": next_hint,
            }
        elif not _final_delivery_attempted(action, checker):
            final_hint = _final_delivery_hint(action, checker)
            decision = {
                "decision": "NEEDS_FIX",
                "rationale": (
                    "Current candidate passed speed, quantitative quality, and qualitative review, "
                    "but no final composed delivery stack has been benchmarked yet. "
                    + rationale_prefix
                ).strip(),
                "next_instructions": _final_delivery_instructions(final_hint),
                "next_action_hint": final_hint,
            }
        else:
            decision = {
                "decision": "ACCEPT",
                "rationale": (
                    str(
                        checker.get(
                            "rationale",
                            "Benchmark is valid and implementation meets speed/quality criteria.",
                        )
                    )
                    + " No obvious compatible unexplored skill remains."
                ).strip(),
            }
    else:
        decision = {
            "decision": "NEEDS_FIX",
            "rationale": (
                str(checker.get("rationale", f"Benchmark command exited with code {exit_code}."))
                + " "
                + rationale_prefix
            ).strip(),
            "next_instructions": "Use checker stdout/stderr, speed, quantitative quality, and visual review evidence to fix or replace this implementation.",
            "next_action_hint": {**selected, "reason": "implementation did not satisfy checker"},
        }

    decision["iteration"] = int(os.environ.get("EAL_ITERATION", "1"))
    decision["evidence"] = evidence
    _write_json(decision_path, decision)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_skill_index(path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {
        "cache": [],
        "quantization": [],
        "sparse_attention": [],
        "compile": [],
    }
    current = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") and stripped not in {
            "## Cache",
            "## Quantization",
            "## Sparse Attention",
            "## Compile",
            "## Optional Helper",
        }:
            current = ""
            continue
        if stripped == "## Cache":
            current = "cache"
            continue
        if stripped == "## Quantization":
            current = "quantization"
            continue
        if stripped == "## Sparse Attention":
            current = "sparse_attention"
            continue
        if stripped in {"## Compile", "## Optional Helper"}:
            current = "compile"
            continue
        match = re.match(r"- `([^`]+)`:", stripped)
        if current and match:
            groups[current].append(match.group(1))
    return {key: value or DEFAULT_SKILLS[key] for key, value in groups.items()}


def _previous_action_hint() -> dict[str, Any]:
    previous_path = Path(os.environ.get("EAL_PREVIOUS_REVIEWER_DECISION", ""))
    if not previous_path.exists():
        return {}
    try:
        previous = _read_json(previous_path)
    except (OSError, json.JSONDecodeError):
        return {}
    hint = previous.get("next_action_hint")
    if not isinstance(hint, dict):
        hint = {}
    hint = dict(hint)
    if previous.get("next_instructions"):
        hint["next_instructions"] = str(previous.get("next_instructions"))
    if previous.get("rationale"):
        hint["rationale"] = str(previous.get("rationale"))
    return hint


def _is_checker_retest_reason(reason: str) -> bool:
    lowered = reason.lower()
    if any(token in lowered for token in ("oom", "out of memory", "executor", "implementation")):
        return False
    return any(
        token in lowered
        for token in (
            "benchmark invalid",
            "benchmark inconclusive",
            "invalid or inconclusive",
            "benchmark pollution",
            "gpu preflight",
            "requires retest",
            "qualitative review must complete",
            "rerun checker",
            "uncontaminated",
        )
    )


def _parameter_blocker_instruction_lines(blockers: list[str]) -> list[str]:
    if not blockers:
        return []
    return [
        "Known invalid parameter blockers for this candidate: " + " | ".join(blockers),
        "Do not retry any exact blocked parameter setting. If the remaining local parameter dimensions are weak or already measured, switch to the next compatible candidate instead.",
    ]


def _before_executor_candidate_override(
    *,
    workdir: Path,
    group: str,
    variant: str,
    skills: dict[str, list[str]],
    previous_metric: dict[str, Any],
    repair_reason: str,
    parameter_blockers: list[str],
) -> dict[str, str]:
    if group != "sparse_attention" or variant != "pisa":
        return {}
    reason_text = repair_reason.lower()
    if any(
        token in reason_text
        for token in (
            "benchmark command failed",
            "benchmark invalid",
            "requires retest",
            "qualitative review must complete",
        )
    ):
        return {}
    visual_failures = _candidate_visual_failure_count(workdir, "sparse_attention", "pisa")
    if "visual" in reason_text and visual_failures >= 3:
        sparse = skills.get("sparse_attention") or []
        if "spargeattn" in sparse:
            return {
                "group": "sparse_attention",
                "variant": "spargeattn",
                "reason": (
                    "PISA has repeated valid-benchmark visual regressions after directed "
                    "repair attempts; switch to spargeattn instead of retrying the same "
                    "PISA path again"
                ),
            }
    summary = previous_metric.get("summary", {}) if isinstance(previous_metric, dict) else {}
    previous_speedup = _float_or_none(summary.get("aggregate_dit_speedup")) if isinstance(summary, dict) else None
    if previous_speedup is None or previous_speedup >= 1.05:
        return {}
    blocker_text = " ".join(parameter_blockers).lower()
    if "block_size=256" not in blocker_text and "block_size 256" not in blocker_text:
        return {}
    sparse = skills.get("sparse_attention") or []
    if "spargeattn" not in sparse:
        return {}
    return {
        "group": "sparse_attention",
        "variant": "spargeattn",
        "reason": (
            "PISA is below the speed target at the last runnable setting, and the attempted "
            "block_size=256 setting is a measured resource blocker; switch to spargeattn "
            "instead of retrying blocked PISA parameters"
        ),
    }


def _candidate_visual_failure_count(workdir: Path, group: str, variant: str) -> int:
    count = 0
    runs_root = workdir / ".eal" / "runs"
    if not runs_root.exists():
        return 0
    for path in sorted(runs_root.glob("**/iter-*/reviewer_decision.json")):
        try:
            decision = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        evidence = decision.get("evidence", {})
        if not isinstance(evidence, dict):
            continue
        selected = evidence.get("selected_candidate", {})
        if not isinstance(selected, dict):
            continue
        if selected.get("group") != group or selected.get("variant") != variant:
            continue
        if evidence.get("benchmark_valid") is not True:
            continue
        qualitative = evidence.get("qualitative", {})
        if not isinstance(qualitative, dict):
            continue
        label = str(qualitative.get("quality_label", "")).lower()
        qualitative_pass = qualitative.get("qualitative_pass")
        if label == "major_regression" or qualitative_pass is False:
            count += 1
    return count


def _selected_candidate(action: dict[str, Any]) -> dict[str, str]:
    context = action.get("model_context") if isinstance(action.get("model_context"), dict) else {}
    selected_group = str(context.get("selected_group", "")).strip()
    selected_variant = str(context.get("selected_variant", "")).strip()
    primary = action.get("optimization_strategy", {})
    if isinstance(primary, dict):
        primary_skill = primary.get("primary_skill", {})
        if isinstance(primary_skill, dict):
            selected_group = selected_group or str(primary_skill.get("group", "")).strip()
            selected_variant = selected_variant or str(primary_skill.get("variant", "")).strip()
    return {
        "group": selected_group or "sparse_attention",
        "variant": selected_variant or "pisa",
    }


def _decision_evidence(
    *,
    checker: dict[str, Any],
    diff_present: bool,
    no_diff_reason: str,
    selected: dict[str, str],
) -> dict[str, Any]:
    qualitative = checker.get("qualitative", {})
    if not isinstance(qualitative, dict):
        qualitative = {}
    overall = qualitative.get("overall", {})
    if not isinstance(overall, dict):
        overall = {}
    return {
        "selected_candidate": selected,
        "git_diff_present": diff_present,
        "no_diff_reason": no_diff_reason,
        "benchmark_valid": checker.get("benchmark_valid"),
        "implementation_valid": checker.get("implementation_valid"),
        "recommendation": checker.get("recommendation"),
        "benchmark_exit_code": _benchmark_exit_code(checker, 1),
        "speed": checker.get("speed", {}),
        "quality": checker.get("quality", {}),
        "qualitative": {
            "status": qualitative.get("status"),
            "required": qualitative.get("required"),
            "qualitative_pass": overall.get("qualitative_pass"),
            "quality_label": overall.get("quality_label"),
            "summary": overall.get("summary"),
        },
    }


def _rationale_prefix(evidence: dict[str, Any]) -> str:
    selected = evidence.get("selected_candidate", {})
    qualitative = evidence.get("qualitative", {})
    parts = [
        f"candidate={selected.get('group')}:{selected.get('variant')}",
        f"benchmark_valid={evidence.get('benchmark_valid')}",
        f"implementation_valid={evidence.get('implementation_valid')}",
        f"recommendation={evidence.get('recommendation')}",
        f"benchmark_exit_code={evidence.get('benchmark_exit_code')}",
    ]
    if qualitative:
        parts.append(
            "visual="
            + str(qualitative.get("quality_label") or qualitative.get("qualitative_pass"))
        )
        summary = str(qualitative.get("summary") or "").strip()
        if summary:
            parts.append(f"visual_summary={summary}")
    if not evidence.get("git_diff_present") and evidence.get("no_diff_reason"):
        parts.append(f"no_diff_reason={evidence.get('no_diff_reason')}")
    return "; ".join(parts) + "."


def _benchmark_exit_code(checker: dict[str, Any], default: Any) -> int:
    artifacts = checker.get("artifacts", {})
    if isinstance(artifacts, dict):
        value = artifacts.get("benchmark_exit_code")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
    if isinstance(default, int):
        return default
    if isinstance(default, str) and default.strip().lstrip("-").isdigit():
        return int(default)
    return 1


def _benchmark_failure_reason(checker: dict[str, Any], benchmark_exit_code: int) -> str:
    text = " ".join(
        part
        for part in (
            str(checker.get("rationale", "")),
            _checker_artifact_tail(checker, "benchmark_stderr"),
        )
        if part
    ).lower()
    if "out of memory" in text or "torch.outofmemoryerror" in text:
        return "benchmark OOM requires executor memory cleanup or a lower-memory candidate"
    if benchmark_exit_code != 0:
        return "benchmark command failed before valid metrics were produced"
    if "metrics.json was not found" in text or "no image rows" in text:
        return "benchmark artifacts were incomplete and require checker retest"
    return "benchmark invalid or inconclusive"


def _checker_artifact_tail(checker: dict[str, Any], key: str, *, max_chars: int = 4000) -> str:
    artifacts = checker.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return ""
    path_value = artifacts.get(key)
    if not isinstance(path_value, str) or not path_value:
        return ""
    path = Path(path_value)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:]


def _benchmark_failure_fix_instructions(selected: dict[str, str], reason: str) -> str:
    group = selected.get("group", "")
    variant = selected.get("variant", "")
    base = [
        f"Fix the `{group}:{variant}` benchmark failure before retesting.",
        "Preserve prompt, seed, resolution, steps, fixed-baseline policy, image metrics, and DiT-only timing semantics.",
        "Do not treat the failed benchmark as speed or quality evidence.",
    ]
    lowered = reason.lower()
    if "block_size" in lowered and "pisa" in lowered:
        base.extend(
            [
                "Restore the last runnable PISA block_size for the same official benchmark shape.",
                "Record the failed block_size setting as a blocked parameter in executor_result.json.",
                "Do not retry the same PISA block_size unless the kernel/resource constraint changes.",
            ]
        )
    elif "oom" in lowered or "memory" in lowered:
        base.extend(
            [
                "Add explicit model/cache cleanup around fixed-baseline, warmup, and measured candidate phases, or reduce the candidate memory footprint while keeping the same benchmark case.",
                "Keep the fixed baseline artifact and rerun only the candidate comparison after memory cleanup is in place.",
            ]
        )
    else:
        base.append("Use checker stderr and missing-artifact evidence to repair the benchmark path, then rerun Checker.")
    return " ".join(base)


def _quality_fix_instructions(selected: dict[str, str], checker: dict[str, Any]) -> str:
    group = selected.get("group", "")
    variant = selected.get("variant", "")
    base = [
        f"Repair `{group}:{variant}` instead of accepting the speedup.",
        "Preserve prompt, seed, resolution, steps, image metrics, and DiT timing semantics.",
        "Use Checker's visual regression notes as hard constraints, not optional commentary.",
    ]
    if group == "sparse_attention" and variant == "pisa":
        base.extend(
            [
                "Reduce PISA damage by increasing allowed attention density, limiting it to safer layers/blocks, or adding dense fallback for visually sensitive calls.",
                "If PISA cannot preserve the robot/book/lamp scene, try a less destructive sparse method such as spargeattn or combine a quality-preserving cache/quantization skill instead.",
            ]
        )
    elif group == "cache":
        base.append("Tighten reuse thresholds or step windows so cached outputs do not alter visible prompt semantics.")
    elif group == "quantization":
        base.append("Keep sensitive modules in higher precision or switch to selective quantization based on measured module speedups.")
    speed = checker.get("speed", {})
    if isinstance(speed, dict) and speed:
        base.append(f"Keep the best measured speed evidence in view: {json.dumps(speed, sort_keys=True)}.")
    return " ".join(base)


def _abnormal_evidence_reason(checker: dict[str, Any]) -> str:
    if bool(checker.get("implementation_valid", False)):
        return ""
    if checker.get("benchmark_valid") is False:
        return ""
    if _speed_below_target_with_passing_quality(checker):
        return ""
    text = _compact_checker_text(checker)
    abnormal_tokens = (
        "abnormal",
        "anomal",
        "implaus",
        "impossible",
        "inconsistent",
        "integrity failed",
        "integrity failure",
        "failed integrity",
        "theoretical",
        "polluted",
        "contaminated",
        "invalid evidence",
        "evidence failed",
    )
    evidence_tokens = (
        "speed",
        "speedup",
        "benchmark",
        "metric",
        "quality",
        "psnr",
        "artifact",
        "implementation evidence",
    )
    if any(token in text for token in abnormal_tokens) and any(
        token in text for token in evidence_tokens
    ):
        return "abnormal candidate evidence requires same-candidate diagnosis and retest"
    return ""


def _compact_checker_text(checker: dict[str, Any]) -> str:
    parts = [
        str(checker.get("recommendation", "")),
        str(checker.get("rationale", "")),
        _safe_json(checker.get("speed", {})),
        _safe_json(checker.get("quality", {})),
        _safe_json(checker.get("qualitative", {})),
        _safe_json(checker.get("artifacts", {})),
    ]
    return " ".join(part for part in parts if part).lower()


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _abnormal_evidence_fix_instructions(
    selected: dict[str, str],
    reason: str,
) -> str:
    group = selected.get("group", "")
    variant = selected.get("variant", "")
    return (
        f"Keep `{group}:{variant}` as the active candidate. Checker rejected abnormal or internally inconsistent evidence: {reason}. "
        "Executor should inspect the candidate implementation, benchmark/instrumentation assumptions, and integration path, then make the smallest fix needed to make the evidence trustworthy. "
        "Do not switch to a different skill just because this run was rejected. After the fix, Checker must rerun the same candidate and validate both speed and benchmark validity; once evidence is valid, continue directed parameter search and compatible skill composition."
    )


def _speed_below_target_with_passing_quality(checker: dict[str, Any]) -> bool:
    if checker.get("benchmark_valid") is not True:
        return False
    qualitative = checker.get("qualitative", {})
    if isinstance(qualitative, dict):
        overall = qualitative.get("overall", {})
        if not isinstance(overall, dict):
            overall = {}
        if qualitative.get("required") and overall.get("qualitative_pass") is not True:
            return False
        if overall.get("qualitative_pass") is False:
            return False
    speed = checker.get("speed", {})
    if not isinstance(speed, dict) or not speed:
        return False
    speed_flags = [
        entry.get("speedup_valid")
        for entry in speed.values()
        if isinstance(entry, dict) and "speedup_valid" in entry
    ]
    if not speed_flags or any(flag is True for flag in speed_flags):
        return False
    if not any(flag is False for flag in speed_flags):
        return False
    quality = checker.get("quality", {})
    if isinstance(quality, dict):
        for entry in quality.values():
            if isinstance(entry, dict) and entry.get("quality_valid") is False:
                return False
    return True


def _parameter_regression_reason(
    workdir: Path,
    action: dict[str, Any],
    checker: dict[str, Any],
    selected: dict[str, str],
) -> str:
    if checker.get("benchmark_valid") is not True:
        return ""
    if not _speed_below_target_with_passing_quality(checker):
        return ""
    variant = selected.get("variant", "")
    previous = action.get("evidence", {})
    if not isinstance(previous, dict):
        return ""
    previous_metric = previous.get("previous_metric", {})
    if not isinstance(previous_metric, dict):
        return ""
    previous_summary = previous_metric.get("summary", {})
    if not isinstance(previous_summary, dict):
        return ""
    previous_speedup = _float_or_none(previous_summary.get("aggregate_dit_speedup"))
    current_record = _variant_checker_record(checker, variant)
    current_speedup = _float_or_none(current_record.get("speedup"))
    history_reason = _historical_parameter_regression_reason(workdir, checker, selected)
    if previous_speedup is None or current_speedup is None:
        return history_reason

    current_reason = _direct_parameter_regression_reason(
        previous_summary,
        checker,
        selected,
        previous_speedup,
        current_speedup,
    )
    return current_reason or history_reason


def _direct_parameter_regression_reason(
    previous_summary: dict[str, Any],
    checker: dict[str, Any],
    selected: dict[str, str],
    previous_speedup: float,
    current_speedup: float,
) -> str:
    variant = selected.get("variant", "")
    current_record = _variant_checker_record(checker, variant)
    speed_regressed = current_speedup < previous_speedup * 0.98
    previous_psnr = _float_or_none(previous_summary.get("mean_psnr"))
    current_psnr = _variant_psnr(checker, variant)
    quality_label = str(current_record.get("quality_label", "")).lower()
    quality_regressed = quality_label in {"minor_regression", "major_regression"}
    if previous_psnr is not None and current_psnr is not None:
        quality_regressed = quality_regressed or current_psnr < previous_psnr - 1.0
    if not (speed_regressed and quality_regressed):
        return ""

    previous_params = previous_summary.get("parameter_settings", {})
    current_params = {}
    artifacts = checker.get("artifacts", {})
    metrics_path = artifacts.get("metrics") if isinstance(artifacts, dict) else ""
    if isinstance(metrics_path, str) and metrics_path:
        current_params = _metrics_parameter_settings(Path(metrics_path), variant)
    parameter_note = ""
    if isinstance(previous_params, dict) and current_params:
        parameter_note = (
            f" previous_params={json.dumps(previous_params, sort_keys=True)}; "
            f"current_params={json.dumps(current_params, sort_keys=True)};"
        )
    return (
        "The last parameter move regressed both speed and quality relative to the previous "
        f"valid setting: previous_speedup={previous_speedup:.3f}x, "
        f"current_speedup={current_speedup:.3f}x.{parameter_note} "
        "Do not continue making this candidate more aggressive; back off toward the last "
        "better setting, try a different parameter dimension, or switch to the next compatible skill."
    )


def _historical_parameter_regression_reason(
    workdir: Path,
    checker: dict[str, Any],
    selected: dict[str, str],
) -> str:
    variant = selected.get("variant", "")
    current = _variant_checker_record(checker, variant)
    current_speedup = _float_or_none(current.get("speedup"))
    if current_speedup is None:
        return ""
    records = [
        record
        for record in _variant_checker_history(workdir, variant)
        if record.get("benchmark_valid") and _float_or_none(record.get("speedup")) is not None
    ]
    if not records:
        return ""
    best = max(records, key=lambda record: float(record.get("speedup")))
    best_speedup = _float_or_none(best.get("speedup"))
    if best_speedup is None or best_speedup <= current_speedup * 1.02:
        return ""
    current_psnr = _variant_psnr(checker, variant)
    best_psnr = _float_or_none(best.get("psnr"))
    current_label = str(current.get("quality_label", "")).lower()
    best_label = str(best.get("quality_label", "")).lower()
    label_regressed = _quality_label_rank(current_label) > _quality_label_rank(best_label)
    psnr_regressed = (
        current_psnr is not None
        and best_psnr is not None
        and current_psnr < best_psnr - 1.0
    )
    if not (label_regressed or psnr_regressed):
        return ""
    return (
        "Current setting is slower and lower quality than a previously measured setting for "
        f"`{variant}`: best_historical_speedup={best_speedup:.3f}x, "
        f"current_speedup={current_speedup:.3f}x. Do not continue making this candidate "
        "more aggressive; back off toward the better historical setting, try a different "
        "parameter dimension, or switch to the next compatible skill."
    )


def _quality_label_rank(label: str) -> int:
    normalized = label.lower().strip()
    if normalized in {"pass", "similar", "same", "visual_pass"}:
        return 0
    if normalized in {"minor_regression", "minor", "slight_regression"}:
        return 1
    if normalized in {"major_regression", "fail", "failed"}:
        return 2
    return 1 if normalized else 0


def _variant_psnr(checker: dict[str, Any], variant: str) -> float | None:
    quality = checker.get("quality", {})
    current_quality = quality.get(variant, {}) if isinstance(quality, dict) else {}
    if not isinstance(current_quality, dict):
        return None
    return _float_or_none(current_quality.get("mean_psnr"))


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _metrics_parameter_settings(path: Path, variant: str) -> dict[str, Any]:
    try:
        metrics = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}
    summary = metrics.get("summary", {})
    if not isinstance(summary, dict):
        return {}
    variants = summary.get("variants", {})
    if isinstance(variants, dict):
        variant_summary = variants.get(variant, {})
        if isinstance(variant_summary, dict):
            settings = variant_summary.get("parameter_settings", {})
            if isinstance(settings, dict):
                return settings
    settings = summary.get("parameter_settings", {})
    if isinstance(settings, dict):
        nested = settings.get(variant)
        if isinstance(nested, dict):
            return nested
        return settings
    return {}


def _parameter_regression_instructions(
    selected: dict[str, str],
    checker: dict[str, Any],
    reason: str,
) -> str:
    group = selected.get("group", "")
    variant = selected.get("variant", "")
    base = [
        f"Do not keep pushing `{group}:{variant}` in the same aggressive direction.",
        reason,
        "Executor should back off toward the last better measured setting or test a different parameter dimension only with a concrete speed rationale.",
        "If the prior and backed-off settings both stay below the speed target, mark this direction as blocked and move to the next compatible skill instead of repeating the same degradation.",
        "Keep prompt, seed, official model hyperparameters, fixed baseline, image metrics, and DiT-only timing unchanged.",
    ]
    if group == "sparse_attention" and variant == "pisa":
        base.append(
            "For PISA specifically, do not lower density further after a slower and visually worse run; restore density toward the previous better setting or try a different sparse-attention candidate such as spargeattn."
        )
    speed = checker.get("speed", {})
    quality = checker.get("quality", {})
    if isinstance(speed, dict) and speed:
        base.append(f"Current speed evidence: {json.dumps(speed, sort_keys=True)}.")
    if isinstance(quality, dict) and quality:
        base.append(f"Current quality evidence: {json.dumps(quality, sort_keys=True)}.")
    return " ".join(base)


def _speed_parameter_search_instructions(
    selected: dict[str, str],
    checker: dict[str, Any],
) -> str:
    group = selected.get("group", "")
    variant = selected.get("variant", "")
    base = [
        f"Keep `{group}:{variant}` active because benchmark validity and visual quality passed.",
        "Do not classify this as benchmark pollution or abnormal evidence.",
        "Continue directed parameter search with an evidence-aware move toward more speed, then rerun Checker against the same fixed baseline.",
    ]
    if group == "sparse_attention" and variant == "pisa":
        base.append(
            "For PISA, do not lower density if lower-density settings were already slower or lower quality. Prefer a different parameter dimension with a concrete rationale, or switch to another sparse-attention candidate such as spargeattn."
        )
    elif group == "cache":
        base.append("For cache/reuse, increase reuse only within the last visually passing window or threshold bracket.")
    elif group == "quantization":
        base.append("For quantization, expand the selective FFN/Linear target set only where visual quality remains stable.")
    speed = checker.get("speed", {})
    quality = checker.get("quality", {})
    if isinstance(speed, dict) and speed:
        base.append(f"Previous speed evidence: {json.dumps(speed, sort_keys=True)}.")
    if isinstance(quality, dict) and quality:
        base.append(f"Previous quality evidence: {json.dumps(quality, sort_keys=True)}.")
    return " ".join(base)


def _pisa_density_direction_blocker(
    workdir: Path,
    selected: dict[str, str],
    current_checker: dict[str, Any],
) -> str:
    if selected.get("group") != "sparse_attention" or selected.get("variant") != "pisa":
        return ""
    if not _speed_below_target_with_passing_quality(current_checker):
        return ""
    current = _variant_checker_record(current_checker, "pisa")
    current_density = _float_or_none(current.get("params", {}).get("density"))
    current_speedup = _float_or_none(current.get("speedup"))
    if current_density is None or current_speedup is None:
        return ""

    records = [
        record
        for record in _variant_checker_history(workdir, "pisa")
        if record.get("benchmark_valid")
    ]
    lower_density_records = []
    for record in records:
        params = record.get("params", {})
        density = _float_or_none(params.get("density")) if isinstance(params, dict) else None
        speedup = _float_or_none(record.get("speedup"))
        if density is None or speedup is None or density >= current_density:
            continue
        lower_density_records.append({**record, "density": density, "speedup": speedup})
    if not lower_density_records:
        return ""

    best_lower = max(lower_density_records, key=lambda record: float(record["speedup"]))
    best_lower_speedup = _float_or_none(best_lower.get("speedup"))
    if best_lower_speedup is None or best_lower_speedup > current_speedup * 0.99:
        return ""

    current_psnr = _float_or_none(current.get("psnr"))
    lower_psnr = _float_or_none(best_lower.get("psnr"))
    current_label = str(current.get("quality_label", "")).lower()
    lower_label = str(best_lower.get("quality_label", "")).lower()
    lower_quality_not_better = (
        _quality_label_rank(lower_label) >= _quality_label_rank(current_label)
    )
    if current_psnr is not None and lower_psnr is not None:
        lower_quality_not_better = lower_quality_not_better or lower_psnr < current_psnr - 0.5
    if not lower_quality_not_better:
        return ""

    tried = sorted({float(record["density"]) for record in lower_density_records})
    return (
        "PISA density direction is already bracketed: the current density="
        f"{current_density:g} setting is still below the speed target, and lower-density "
        f"settings {tried} were slower or lower quality. Do not lower PISA density again; "
        "switch to another sparse-attention candidate or a non-density parameter dimension "
        "with a concrete speed rationale."
    )


def _pisa_density_blocked_next_hint(
    action: dict[str, Any],
    reason: str,
) -> dict[str, str]:
    context = action.get("model_context") if isinstance(action.get("model_context"), dict) else {}
    strategy = context.get("optimization_strategy", {})
    skills = strategy.get("available_skill_groups") if isinstance(strategy, dict) else {}
    if not isinstance(skills, dict):
        skills = {}
    sparse = [str(item) for item in skills.get("sparse_attention", []) if isinstance(item, str)]
    if "spargeattn" in sparse:
        return {
            "group": "sparse_attention",
            "variant": "spargeattn",
            "reason": reason,
        }
    return {
        "group": "sparse_attention",
        "variant": "pisa",
        "reason": reason,
    }


def _pisa_density_blocked_instructions(
    hint: dict[str, str],
    checker: dict[str, Any],
    reason: str,
) -> str:
    target = f"{hint.get('group')}:{hint.get('variant')}"
    base = [
        reason,
        "Do not set PISA density to a lower value such as 0.25 or 0.125 again in the next iteration.",
        f"Use `{target}` as the next candidate unless Executor can justify a non-density PISA parameter change with concrete speed rationale.",
        "Keep prompt, seed, official model hyperparameters, fixed baseline, image metrics, and DiT-only timing unchanged.",
    ]
    speed = checker.get("speed", {})
    quality = checker.get("quality", {})
    if isinstance(speed, dict) and speed:
        base.append(f"Current speed evidence: {json.dumps(speed, sort_keys=True)}.")
    if isinstance(quality, dict) and quality:
        base.append(f"Current quality evidence: {json.dumps(quality, sort_keys=True)}.")
    return " ".join(base)


def _parameter_frontier_blocker(
    workdir: Path,
    selected: dict[str, str],
    current_checker: dict[str, Any],
) -> str:
    group = selected.get("group", "")
    variant = selected.get("variant", "")
    if not variant:
        return ""
    records = _variant_checker_history(workdir, variant)
    current_record = _variant_checker_record(current_checker, variant)
    if current_record:
        records.append(current_record)
    valid = [record for record in records if record["benchmark_valid"]]
    if len(valid) < 4:
        return ""
    if any(record["implementation_valid"] for record in valid):
        return ""
    quality_passing_slow = any(
        record["qualitative_pass"] is True and record["speedup_valid"] is False
        for record in valid
    )
    quality_failed = any(
        record["qualitative_pass"] is False or record["quality_label"] == "major_regression"
        for record in valid
    )
    if not (quality_passing_slow and quality_failed):
        return ""
    speedups = [
        record["speedup"]
        for record in valid
        if isinstance(record.get("speedup"), int | float)
    ]
    best_speedup = max(speedups) if speedups else None
    best_text = f" best observed speedup={best_speedup:.3f}x;" if best_speedup is not None else ""
    return (
        f"{group}:{variant} parameter frontier is bracketed:{best_text} "
        "quality-passing settings stayed below the speed target, while more aggressive settings "
        "failed required visual quality. Switch to the next compatible skill instead of continuing "
        "same-candidate parameter search."
    )


def _variant_checker_history(workdir: Path, variant: str) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    runs_root = workdir / ".eal" / "runs"
    if not runs_root.exists():
        return history
    for path in sorted(runs_root.glob("**/iter-*/checker_result.json")):
        try:
            checker = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        record = _variant_checker_record(checker, variant)
        if record:
            record["path"] = str(path)
            history.append(record)
    return history


def _variant_checker_record(checker: dict[str, Any], variant: str) -> dict[str, Any]:
    speed = checker.get("speed", {})
    if not isinstance(speed, dict) or variant not in speed:
        return {}
    speed_entry = speed.get(variant)
    if not isinstance(speed_entry, dict):
        return {}
    qualitative = checker.get("qualitative", {})
    overall = qualitative.get("overall", {}) if isinstance(qualitative, dict) else {}
    if not isinstance(overall, dict):
        overall = {}
    return {
        "benchmark_valid": checker.get("benchmark_valid") is True,
        "implementation_valid": checker.get("implementation_valid") is True,
        "qualitative_pass": overall.get("qualitative_pass"),
        "quality_label": str(overall.get("quality_label", "")),
        "speedup_valid": speed_entry.get("speedup_valid"),
        "speedup": speed_entry.get("aggregate_dit_speedup"),
        "psnr": _variant_psnr(checker, variant),
        "params": _checker_parameter_settings(checker, variant),
    }


def _checker_parameter_settings(checker: dict[str, Any], variant: str) -> dict[str, Any]:
    artifacts = checker.get("artifacts", {})
    if isinstance(artifacts, dict):
        metrics_path = artifacts.get("metrics")
        if isinstance(metrics_path, str) and metrics_path:
            settings = _metrics_parameter_settings(Path(metrics_path), variant)
            if settings:
                return settings
        settings = artifacts.get("parameter_settings")
        if isinstance(settings, dict):
            nested = settings.get(variant)
            if isinstance(nested, dict):
                return nested
            if settings:
                return settings
    settings = checker.get("parameter_settings")
    if isinstance(settings, dict):
        nested = settings.get(variant)
        if isinstance(nested, dict):
            return nested
        return settings
    return {}


def _invalid_parameter_blockers(workdir: Path, group: str, variant: str) -> list[str]:
    blockers: list[str] = []
    runs_root = workdir / ".eal" / "runs"
    if not runs_root.exists():
        return blockers
    for executor_path in sorted(runs_root.glob("**/iter-*/executor_result.json")):
        try:
            executor = _read_json(executor_path)
        except (OSError, json.JSONDecodeError):
            continue
        blockers.extend(_executor_blocked_parameter_settings(executor, group, variant))
        checker_path = executor_path.with_name("checker_result.json")
        if not checker_path.exists():
            continue
        try:
            checker = _read_json(checker_path)
        except (OSError, json.JSONDecodeError):
            continue
        if checker.get("benchmark_valid") is not False:
            continue
        failure_text = _resource_failure_text(checker)
        if not failure_text:
            continue
        params = _executor_candidate_parameters(executor, group, variant)
        if params:
            blockers.append(_format_parameter_blocker(variant, params, failure_text))
    return sorted(dict.fromkeys(blockers))


def _executor_blocked_parameter_settings(
    executor: dict[str, Any],
    group: str,
    variant: str,
) -> list[str]:
    artifacts = executor.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return []
    final_stack = artifacts.get("final_stack", {})
    if not isinstance(final_stack, dict):
        return []
    blocked = final_stack.get("blocked_parameter_settings", [])
    if not isinstance(blocked, list):
        return []
    results: list[str] = []
    for item in blocked:
        if not isinstance(item, dict):
            continue
        skill_id = str(item.get("skill_id", "")).strip()
        role = str(item.get("role", "")).strip()
        if skill_id != variant:
            continue
        if group == "sparse_attention" and role not in {"", "attention"}:
            continue
        params = item.get("parameters", {})
        if not isinstance(params, dict):
            params = {}
        blocker = str(item.get("blocker", "invalid parameter setting")).strip()
        results.append(_format_parameter_blocker(variant, params, blocker))
    return results


def _executor_candidate_parameters(
    executor: dict[str, Any],
    group: str,
    variant: str,
) -> dict[str, Any]:
    artifacts = executor.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return {}
    settings = artifacts.get("parameter_settings", {})
    if not isinstance(settings, dict):
        return {}
    if group == "sparse_attention":
        sparse = settings.get("sparse_attention", {})
        if isinstance(sparse, dict) and sparse.get("variant") == variant:
            return {
                key: sparse[key]
                for key in (
                    "apply_to",
                    "density",
                    "block_size",
                    "dense_initial_steps",
                    "topk",
                    "min_tokens",
                    "max_tokens",
                )
                if key in sparse
            }
    nested = settings.get(variant)
    return nested if isinstance(nested, dict) else {}


def _resource_failure_text(checker: dict[str, Any]) -> str:
    text = " ".join(
        part
        for part in (
            str(checker.get("rationale", "")),
            _checker_artifact_tail(checker, "benchmark_stderr"),
        )
        if part
    ).lower()
    if "outofresources" in text or "out of resource" in text:
        if "shared memory" in text:
            return "Triton shared-memory OutOfResources"
        return "Triton OutOfResources"
    if "out of memory" in text or "torch.outofmemoryerror" in text:
        return "CUDA out of memory"
    return ""


def _format_parameter_blocker(
    variant: str,
    params: dict[str, Any],
    blocker: str,
) -> str:
    ordered_keys = [
        "apply_to",
        "density",
        "block_size",
        "dense_initial_steps",
        "topk",
        "min_tokens",
        "max_tokens",
    ]
    parts = [f"{key}={params[key]}" for key in ordered_keys if key in params]
    setting = ", ".join(parts) if parts else "parameters=unknown"
    return f"{variant} {setting}: {blocker}"


def _parameter_resource_failure_reason(
    selected: dict[str, str],
    checker: dict[str, Any],
    executor: dict[str, Any],
) -> str:
    if selected.get("group") != "sparse_attention" or selected.get("variant") != "pisa":
        return ""
    failure_text = _resource_failure_text(checker)
    if not failure_text:
        return ""
    params = _executor_candidate_parameters(executor, "sparse_attention", "pisa")
    if failure_text == "CUDA out of memory" and not params:
        return ""
    block_size = params.get("block_size")
    if block_size:
        return (
            f"PISA parameter setting block_size={block_size} failed with {failure_text} "
            "before valid metrics were produced. Restore the last runnable block_size, "
            "record this setting as blocked, and do not retry the same block_size unless "
            "the kernel/resource constraint changes."
        )
    return (
        f"PISA candidate failed with {failure_text} before valid metrics were produced. "
        "Reduce the candidate resource footprint, record the failed setting as blocked, and retest."
    )


def _exploration_instructions(hint: dict[str, Any]) -> str:
    group = str(hint.get("group", "")).strip()
    variant = str(hint.get("variant", "")).strip()
    return (
        f"Continue optimization with the next compatible candidate `{group}:{variant}`. "
        "Integrate it as part of the broader skill stack, preserve all benchmark semantics, "
        "and only accept after speed, quantitative quality, required visual review, and "
        "remaining compatible skill-space checks all pass."
    )


def _final_delivery_attempted(action: dict[str, Any], checker: dict[str, Any]) -> bool:
    variants = _checker_variants(checker)
    selected = _selected_candidate(action)
    variants.add(selected.get("variant", ""))
    return any(
        token in variant.lower()
        for variant in variants
        for token in ("final", "stack", "compile", "+")
    )


def _final_delivery_hint(action: dict[str, Any], checker: dict[str, Any]) -> dict[str, Any]:
    selected = _selected_candidate(action)
    context = action.get("model_context") if isinstance(action.get("model_context"), dict) else {}
    strategy = context.get("optimization_strategy", {})
    skills = strategy.get("available_skill_groups") if isinstance(strategy, dict) else {}
    if not isinstance(skills, dict):
        skills = {}
    quant_variants = [str(item) for item in skills.get("quantization", []) if isinstance(item, str)]
    preferred_quant = next(
        (item for item in ("selective_torchao_nvfp4", "selective_torchao_fp8", "selective_torchao_mxfp8") if item in quant_variants),
        selected.get("variant", "selective_torchao_nvfp4"),
    )
    return {
        "group": "quantization" if quant_variants else selected.get("group", "quantization"),
        "variant": preferred_quant,
        "reason": (
            "build final composed delivery stack from best measured components; include or explicitly rule out "
            "tuned sparse attention, selective FFN/Linear quantization, torch_compile, and compatible cache/reuse"
        ),
    }


def _final_delivery_instructions(hint: dict[str, Any]) -> str:
    return (
        "Build and benchmark a final composed Flux delivery stack. Use the best measured compatible sparse-attention "
        "setting, selective FFN/Linear quantization setting, torch_compile if compatible, and any compatible cache/reuse "
        "component. Do not accept a catalog of isolated skill results. Record exact parameters, included components, "
        "excluded components with measured blockers, fixed-baseline benchmark artifact, and visual review evidence."
    )


def _next_candidate_hint(
    workdir: Path,
    action: dict[str, Any],
    checker: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    context = action.get("model_context") if isinstance(action.get("model_context"), dict) else {}
    skills = context.get("optimization_strategy", {}).get("available_skill_groups")
    if not isinstance(skills, dict):
        skills = context.get("available_skills")
    if not isinstance(skills, dict):
        strategy = action.get("optimization_strategy", {})
        skills = strategy.get("available_skill_groups") if isinstance(strategy, dict) else {}
    normalized = {
        key: [str(item) for item in value]
        for key, value in (skills if isinstance(skills, dict) else DEFAULT_SKILLS).items()
        if isinstance(value, list)
    }
    for key, value in DEFAULT_SKILLS.items():
        normalized.setdefault(key, value)
    selected = _selected_candidate(action)
    explored = _explored_variants(workdir) | _checker_variants(checker) | {selected["variant"]}
    candidates = _candidate_sequence(normalized, selected)
    for candidate in candidates:
        if candidate["variant"] not in explored:
            return {**candidate, "reason": reason}
    return {}


def _candidate_sequence(
    skills: dict[str, list[str]],
    selected: dict[str, str],
) -> list[dict[str, str]]:
    ordered_groups = [
        selected.get("group", "sparse_attention"),
        "cache",
        "quantization",
        "sparse_attention",
    ]
    sequence: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for group in ordered_groups:
        for variant in skills.get(group, []):
            key = (group, variant)
            if key in seen:
                continue
            seen.add(key)
            sequence.append({"group": group, "variant": variant})
    return sequence


def _explored_variants(workdir: Path) -> set[str]:
    explored: set[str] = set()
    for path in (workdir / "benchmark" / "artifacts").glob("*/metrics.json"):
        try:
            data = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        rows = data.get("rows", [])
        if isinstance(rows, list):
            explored.update(str(row.get("variant")) for row in rows if isinstance(row, dict) and row.get("variant"))
    return explored


def _checker_variants(checker: dict[str, Any]) -> set[str]:
    variants: set[str] = set()
    for values in (checker.get("speed", {}), checker.get("quality", {})):
        if isinstance(values, dict):
            variants.update(str(key) for key in values)
    return variants


def _choose_group(requested: str, text: str, skills: dict[str, list[str]]) -> str:
    if requested in skills:
        return requested
    if "quant" in text or "fp8" in text or "fp4" in text:
        return "quantization"
    if "cache" in text or "reuse" in text or "tea" in text:
        return "cache"
    if "sparse" in text or "attention" in text or "pisa" in text or "sparge" in text:
        return "sparse_attention"
    return "sparse_attention"


def _choose_variant(requested: str, group: str, skills: dict[str, list[str]]) -> str:
    available = skills.get(group) or DEFAULT_SKILLS[group]
    if requested in available:
        return requested
    preferred = {
        "cache": "periodic_reuse",
        "quantization": "selective_torchao_nvfp4",
        "sparse_attention": "pisa",
    }[group]
    return preferred if preferred in available else available[0]


def _parameter_search_policy(group: str, variant: str) -> dict[str, Any]:
    return {
        "mode": "directed_tradeoff_search",
        "primary_candidate": {"group": group, "variant": variant},
        "not_a_grid_search": True,
        "rule": (
            "Explore the local parameter frontier with evidence-aware moves, not exhaustive "
            "enumeration and not one setting per skill."
        ),
        "adaptation_rules": [
            "If speedup is below target and quantitative/visual quality is strong, move to a more aggressive setting.",
            "If speedup is strong but PSNR or multimodal visual quality regresses, back off toward the last passing setting or try an intermediate setting.",
            "If a more aggressive setting is slower and quality also regresses, do not keep pushing that direction; back off, try a different parameter dimension, or switch candidates.",
            "For PISA, if lower density settings have already been measured slower or lower quality, do not lower density again; switch density direction, change another parameter, or switch candidates.",
            "If both speed and quality pass, test compatibility with the next stack component instead of declaring the skill family done.",
            "If a setting fails because the benchmark is invalid, fix/retest benchmark validity before drawing parameter conclusions.",
        ],
        "search_dimensions": {
            "sparse_attention": [
                "pisa_density",
                "pisa_block_size",
                "pisa_dense_initial_steps",
                "sparge_topk",
                "apply_to",
                "periodic_reuse composition knobs",
            ],
            "cache": [
                "reuse_interval",
                "warmup_steps",
                "max_skip_steps",
                "similarity/delta/EMA thresholds",
            ],
            "quantization": [
                "recipe",
                "min_speedup",
                "skip_modules",
                "selected module families such as FFN/Linear projections",
                "high-precision keep list",
            ],
            "compile": [
                "compile scope: whole DiT, hot blocks, FFN/Linear path, attention path",
                "backend and mode",
                "warmup count needed to exclude compile overhead",
            ],
        },
        "executor_requirement": (
            "When changing a candidate, state the exact parameter values and why the move is "
            "more aggressive, more conservative, or an intermediate tradeoff."
        ),
        "checker_requirement": (
            "Metrics should preserve candidate parameter metadata so Reviewer can compare "
            "neighboring settings and choose the next tradeoff move."
        ),
        "reviewer_requirement": (
            "Do not consider a skill family exhausted until the useful parameter frontier is "
            "bracketed by at least one too-conservative or too-slow setting and one "
            "too-aggressive or quality-regressing setting, or until a measured blocker exists."
        ),
    }


def _final_delivery_policy(skills: dict[str, list[str]]) -> dict[str, Any]:
    compile_skills = skills.get("compile") or skills.get("compilation") or ["torch_compile"]
    return {
        "deliverable": "one composed Flux optimization stack",
        "not_deliverable": "a report that individual skill ids were sampled once",
        "target_stack_roles": [
            {
                "role": "attention",
                "examples": ["tuned pisa", "tuned spargeattn"],
                "required_decision": "include the best valid sparse attention setting or explain the measured blocker",
            },
            {
                "role": "ffn_linear_quantization",
                "examples": [
                    "selective_torchao_fp8",
                    "selective_torchao_mxfp8",
                    "selective_torchao_nvfp4",
                ],
                "required_decision": "include the best valid selective FFN/Linear quant setting or explain the measured blocker",
            },
            {
                "role": "compile",
                "examples": compile_skills,
                "required_decision": "benchmark compile composition or explain why compile is incompatible/unavailable",
            },
            {
                "role": "cache_reuse",
                "examples": ["periodic_reuse", "similarity_reuse", "delta_reuse", "ema_reuse"],
                "required_decision": "include only if compatible with visual quality and the fixed-baseline benchmark",
            },
        ],
        "acceptance_gate": (
            "ACCEPT only after the composed stack is implemented, benchmarked against the "
            "fixed baseline, visually reviewed, and compared against the best individual "
            "components. If compile or another role is excluded, Reviewer must record the "
            "specific measured failure or incompatibility."
        ),
        "handoff_requirement": (
            "The final accepted Executor result should identify the exact stack, parameter "
            "values, changed files, benchmark artifact, and any excluded components with reasons."
        ),
    }


def _official_model_config() -> dict[str, Any]:
    return dict(OFFICIAL_FLUX_BENCHMARK_CONFIG)


def _official_config_instruction() -> str:
    config = OFFICIAL_FLUX_BENCHMARK_CONFIG
    return (
        "Default to official Flux model hyperparameters unless the user explicitly "
        f"overrides them: width={config['width']}, height={config['height']}, "
        f"steps={config['steps']}, guidance_scale={config['guidance_scale']}, "
        f"max_sequence_length={config['max_sequence_length']} for this FLUX.1 "
        "schnell benchmark, and sampler/scheduler/time_shift from the official "
        "model or workflow defaults. Do not use legacy 512x512 smoke settings as "
        "acceptance evidence."
    )


def _baseline_policy() -> dict[str, Any]:
    return {
        "scope": "run_level_fixed_baseline",
        "artifact_name": BASELINE_ARTIFACT_NAME,
        "artifact_path": f"benchmark/artifacts/{BASELINE_ARTIFACT_NAME}",
        "fixed_config": {
            "model_family": OFFICIAL_FLUX_BENCHMARK_CONFIG["model_family"],
            "config_source": OFFICIAL_FLUX_BENCHMARK_CONFIG["source"],
            "prompts": "PROMPTS from benchmark.flux_schnell_benchmark",
            "seeds": "SEEDS from benchmark.flux_schnell_benchmark",
            "width": OFFICIAL_FLUX_BENCHMARK_CONFIG["width"],
            "height": OFFICIAL_FLUX_BENCHMARK_CONFIG["height"],
            "steps": OFFICIAL_FLUX_BENCHMARK_CONFIG["steps"],
            "guidance": OFFICIAL_FLUX_BENCHMARK_CONFIG["guidance"],
            "guidance_scale": OFFICIAL_FLUX_BENCHMARK_CONFIG["guidance_scale"],
            "cfg": OFFICIAL_FLUX_BENCHMARK_CONFIG["cfg"],
            "max_sequence_length": OFFICIAL_FLUX_BENCHMARK_CONFIG["max_sequence_length"],
            "sampler": OFFICIAL_FLUX_BENCHMARK_CONFIG["sampler"],
            "scheduler": OFFICIAL_FLUX_BENCHMARK_CONFIG["scheduler"],
            "time_shift": OFFICIAL_FLUX_BENCHMARK_CONFIG["time_shift"],
            "timing_scope": "dit_denoise_only",
            "warmup_scope": "before_each_measured_workflow",
        },
        "rule": (
            "Create the baseline once for the run and reuse its timing, image, "
            "and metadata for every candidate comparison."
        ),
        "rerun_baseline_only_if": [
            "baseline artifact is missing",
            "baseline artifact is corrupt or incomplete",
            "prompt, seed, official model hyperparameters, model, or timing scope changed",
            "Reviewer explicitly requests a fresh baseline",
        ],
        "checker_requirement": (
            "If a valid fixed baseline exists but the benchmark compares a candidate "
            "against a newly measured per-iteration baseline, report "
            "benchmark_valid=false and NEEDS_RETEST. If metrics do not match the "
            "fixed official width/height/steps "
            "and there is no explicit user override, report benchmark_valid=false."
        ),
    }


def _source_edit_policy(
    repair_reason: str,
    *,
    authorize_benchmark_for_baseline: bool = False,
) -> dict[str, Any]:
    reason_text = repair_reason.lower()
    authorization = {
        "efficient_skill": False,
        "benchmark": False,
        "reasons": [],
        "rationale": "Default to canvas, config, experiment wiring, and thin integration changes.",
    }
    if authorize_benchmark_for_baseline:
        authorization["benchmark"] = True
        authorization["reasons"].append("benchmark_validity")
        authorization["rationale"] = (
            "Benchmark edits are authorized only to support run-level fixed baseline "
            "reuse and related benchmark-validity checks."
        )
    if _mentions_benchmark_validity(reason_text):
        authorization["benchmark"] = True
        if "benchmark_validity" not in authorization["reasons"]:
            authorization["reasons"].append("benchmark_validity")
    if _mentions_reusable_skill_need(reason_text):
        authorization["efficient_skill"] = True
        authorization["reasons"].append("reusable_skill_behavior")
    if authorization["reasons"] and not authorize_benchmark_for_baseline:
        authorization["rationale"] = (
            "Previous Reviewer hint indicates a restricted edit may be necessary; "
            "Executor should still prefer the narrowest compatible change."
        )
    return {
        "default_edit_surface": [
            "draft_canvas/",
            "comfy_extras/",
            "workflow/config/experiment wiring",
        ],
        "restricted_edit_surface": [
            RESTRICTED_EDIT_PREFIXES["efficient_skill"],
            RESTRICTED_EDIT_PREFIXES["benchmark"],
        ],
        "allowed_restricted_reasons": ALLOWED_RESTRICTED_REASONS,
        "current_authorization": authorization,
        "executor_requirement": (
            "Do not modify a restricted surface unless current_authorization enables "
            "that surface. If authorization is missing, leave it unchanged and explain "
            "the blocked change in executor_result.json."
        ),
    }


def _mentions_benchmark_validity(reason_text: str) -> bool:
    return any(
        token in reason_text
        for token in (
            "benchmark invalid",
            "benchmark pollution",
            "benchmark_valid",
            "polluted",
            "retest",
            "uncontaminated",
            "artifact",
            "cannot measure",
            "oom",
        )
    )


def _mentions_reusable_skill_need(reason_text: str) -> bool:
    return any(
        token in reason_text
        for token in (
            "implementation did not satisfy checker",
            "repair executor failure",
            "repair visual quality regression",
            "current candidate was rejected",
            "missing skill",
            "skill interface",
            "reusable skill",
        )
    )


def _authorization_summary(policy: dict[str, Any]) -> str:
    authorization = policy.get("current_authorization", {})
    if not isinstance(authorization, dict):
        authorization = {}
    reasons = authorization.get("reasons")
    reason_text = ", ".join(str(reason) for reason in reasons) if isinstance(reasons, list) else ""
    return (
        "Current restricted edit authorization: "
        f"efficient_skill={bool(authorization.get('efficient_skill'))}, "
        f"benchmark={bool(authorization.get('benchmark'))}, "
        f"reasons={reason_text or 'none'}."
    )


def _combined_edit_areas(primary_group: str) -> list[str]:
    paths: list[str] = []
    for group in (primary_group, "cache", "quantization", "sparse_attention"):
        for path in EDIT_AREAS.get(group, []):
            if path not in paths:
                paths.append(path)
    return paths


def _default_edit_areas(primary_group: str) -> list[str]:
    return [
        path
        for path in _combined_edit_areas(primary_group)
        if not _restricted_surface(path)
    ]


def _restricted_edit_areas(primary_group: str) -> list[str]:
    return [
        path
        for path in _combined_edit_areas(primary_group)
        if _restricted_surface(path)
    ]


def _authorized_edit_areas(
    default_paths: list[str],
    restricted_paths: list[str],
    policy: dict[str, Any],
) -> list[str]:
    authorized = list(default_paths)
    authorization = policy.get("current_authorization", {})
    if not isinstance(authorization, dict):
        authorization = {}
    for path in restricted_paths:
        if path.startswith(RESTRICTED_EDIT_PREFIXES["efficient_skill"]) and authorization.get(
            "efficient_skill"
        ):
            authorized.append(path)
        elif path.startswith(RESTRICTED_EDIT_PREFIXES["benchmark"]) and authorization.get(
            "benchmark"
        ):
            authorized.append(path)
    return authorized


def _restricted_surface(path: str) -> str:
    for surface, prefix in RESTRICTED_EDIT_PREFIXES.items():
        if path.startswith(prefix):
            return surface
    return ""


def _checker_command(
    group: str,
    benchmark: str,
    variant: str,
    artifact_name: str,
) -> str:
    config = OFFICIAL_FLUX_BENCHMARK_CONFIG
    command = (
        f"conda run -n auto_deploy_flux_eff python -B {benchmark} "
        f"--no-download --warmup-runs 1 --max-cases 1 "
        f"--width {config['width']} --height {config['height']} --steps {config['steps']} "
        f"--guidance-scale {config['guidance_scale']} "
        f"--max-sequence-length {config['max_sequence_length']} "
        f"--artifact-name {artifact_name} "
        f"--baseline-artifact-name {BASELINE_ARTIFACT_NAME}"
    )
    return command


def _find_previous_metrics(workdir: Path, variant: str) -> dict[str, Any]:
    candidates = sorted(
        (workdir / "benchmark" / "artifacts").glob("*/metrics.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            data = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        summary = data.get("summary", {})
        variants = summary.get("variants", {})
        if isinstance(variants, dict) and variant in variants:
            return {
                "artifact": str(path.parent),
                "variant": variant,
                "summary": variants[variant],
            }
    return {}


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return _truncate(result.stderr.strip())
    return _truncate(result.stdout.strip())


def _truncate(text: str, max_lines: int = 80, max_chars: int = 12000) -> str:
    lines = text.splitlines()
    clipped = "\n".join(lines[:max_lines])
    if len(lines) > max_lines:
        clipped += f"\n... truncated {len(lines) - max_lines} more lines ..."
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars] + "\n... truncated by character limit ..."
    return clipped


if __name__ == "__main__":
    main()
