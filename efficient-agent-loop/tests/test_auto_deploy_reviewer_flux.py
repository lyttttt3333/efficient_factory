import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEWER = REPO_ROOT / "examples" / "auto_deploy" / "reviewer_flux.py"


def test_flux_reviewer_visual_regression_overrides_explained_no_diff(tmp_path):
    decision = _run_after_checker(
        tmp_path,
        diff_text="",
        executor={
            "status": "completed",
            "summary": "verification-only",
            "no_diff_reason": "PISA was already wired and only needed checking.",
            "no_diff_reason_source": "executor",
            "command": {"exit_code": 0},
        },
        checker={
            "benchmark_valid": True,
            "implementation_valid": False,
            "git_diff_unchanged": True,
            "recommendation": "NEEDS_FIX",
            "speed": {"pisa": {"aggregate_dit_speedup": 6.46}},
            "quality": {"pisa": {"mean_psnr": 12.98}},
            "qualitative": {
                "status": "completed",
                "required": True,
                "overall": {
                    "qualitative_pass": False,
                    "quality_label": "major_regression",
                    "summary": "At least one candidate image has a qualitative regression.",
                },
            },
            "command": {"exit_code": 0},
        },
    )

    assert decision["decision"] == "NEEDS_FIX"
    assert "qualitative review failed" in decision["rationale"]
    assert decision["evidence"]["git_diff_present"] is False
    assert decision["next_action_hint"]["variant"] == "pisa"


def test_flux_reviewer_continues_when_candidate_space_remains(tmp_path):
    decision = _run_after_checker(
        tmp_path,
        diff_text="diff --git a/x b/x\n",
        executor={
            "status": "completed",
            "summary": "changed implementation",
            "command": {"exit_code": 0},
        },
        checker={
            "benchmark_valid": True,
            "implementation_valid": True,
            "git_diff_unchanged": True,
            "recommendation": "ACCEPT",
            "speed": {"pisa": {"aggregate_dit_speedup": 1.2}},
            "quality": {"pisa": {"mean_psnr": 35.0}},
            "qualitative": {
                "status": "completed",
                "required": True,
                "overall": {
                    "qualitative_pass": True,
                    "quality_label": "pass",
                    "summary": "Candidate image is visually comparable to baseline.",
                },
            },
            "command": {"exit_code": 0},
        },
    )

    assert decision["decision"] == "NEEDS_FIX"
    assert "space is not exhausted" in decision["rationale"]
    assert decision["next_action_hint"]["group"] == "sparse_attention"
    assert decision["next_action_hint"]["variant"] == "spargeattn"


def test_flux_reviewer_needs_fix_when_visual_passes_but_speed_fails(tmp_path):
    decision = _run_after_checker(
        tmp_path,
        diff_text="diff --git a/x b/x\n",
        executor={
            "status": "completed",
            "summary": "changed implementation",
            "command": {"exit_code": 0},
        },
        checker={
            "benchmark_valid": True,
            "implementation_valid": False,
            "git_diff_unchanged": True,
            "recommendation": "NEEDS_FIX",
            "speed": {"pisa": {"aggregate_dit_speedup": 0.9868, "speedup_valid": False}},
            "quality": {"pisa": {"mean_psnr": 24.64, "quality_valid": True}},
            "qualitative": {
                "status": "completed",
                "required": True,
                "overall": {
                    "qualitative_pass": True,
                    "quality_label": "minor_regression",
                    "summary": "All reviewed image pairs passed qualitative visual inspection.",
                    "recommendation": "visual_pass",
                    "recommendation_scope": "visual_quality_only",
                },
            },
            "command": {"exit_code": 0},
        },
    )

    assert decision["decision"] == "NEEDS_FIX"
    assert decision["evidence"]["speed"]["pisa"]["speedup_valid"] is False
    assert "speed below target" in decision["next_action_hint"]["reason"]
    assert "directed parameter search" in decision["next_instructions"]


def test_flux_reviewer_backs_off_after_aggressive_parameter_regresses(tmp_path):
    decision = _run_after_checker(
        tmp_path,
        diff_text="diff --git a/x b/x\n",
        executor={
            "status": "completed",
            "summary": "changed implementation",
            "command": {"exit_code": 0},
        },
        checker={
            "benchmark_valid": True,
            "implementation_valid": False,
            "git_diff_unchanged": True,
            "recommendation": "NEEDS_FIX",
            "speed": {
                "pisa": {
                    "aggregate_dit_speedup": 0.8525,
                    "speedup_valid": False,
                }
            },
            "quality": {
                "pisa": {
                    "mean_psnr": 14.73,
                    "quality_valid": True,
                }
            },
            "qualitative": {
                "status": "completed",
                "required": True,
                "overall": {
                    "qualitative_pass": True,
                    "quality_label": "minor_regression",
                    "summary": "All reviewed image pairs passed qualitative visual inspection.",
                    "recommendation": "visual_pass",
                    "recommendation_scope": "visual_quality_only",
                },
            },
            "command": {"exit_code": 0},
        },
        action={
            "model_context": {
                "selected_group": "sparse_attention",
                "selected_variant": "pisa",
                "optimization_strategy": {
                    "available_skill_groups": {
                        "cache": ["periodic_reuse"],
                        "quantization": ["selective_torchao_nvfp4"],
                        "sparse_attention": ["pisa", "spargeattn"],
                    }
                },
            },
            "optimization_strategy": {
                "primary_skill": {"group": "sparse_attention", "variant": "pisa"}
            },
            "evidence": {
                "previous_metric": {
                    "summary": {
                        "aggregate_dit_speedup": 1.0064,
                        "mean_psnr": 17.74,
                        "parameter_settings": {
                            "apply_to": "single",
                            "density": 0.5,
                            "block_size": 128,
                            "dense_initial_steps": 0,
                        },
                    },
                    "variant": "pisa",
                }
            },
        },
    )

    assert decision["decision"] == "NEEDS_FIX"
    assert "regressed both speed and quality" in decision["rationale"]
    assert "Do not keep pushing" in decision["next_instructions"]
    assert "do not lower density further" in decision["next_instructions"]
    assert "regressed both speed and quality" in decision["next_action_hint"]["reason"]


def test_flux_reviewer_uses_history_to_avoid_repeating_bad_aggressive_setting(tmp_path):
    history = [
        _checker_history(
            "pisa",
            speedup=1.0064,
            speedup_valid=False,
            qualitative_pass=True,
            quality_label="similar",
            psnr=17.74,
        ),
        _checker_history(
            "pisa",
            speedup=0.8525,
            speedup_valid=False,
            qualitative_pass=True,
            quality_label="minor_regression",
            psnr=14.73,
        ),
    ]
    current = _checker_history(
        "pisa",
        speedup=0.85,
        speedup_valid=False,
        qualitative_pass=True,
        quality_label="minor_regression",
        psnr=14.7,
    )
    current.update(
        {
            "recommendation": "NEEDS_FIX",
            "command": {"exit_code": 0},
            "git_diff_unchanged": True,
        }
    )

    decision = _run_after_checker(
        tmp_path,
        diff_text="diff --git a/x b/x\n",
        executor={
            "status": "completed",
            "summary": "changed implementation",
            "command": {"exit_code": 0},
        },
        checker=current,
        history_checkers=history,
    )

    assert decision["decision"] == "NEEDS_FIX"
    assert "previously measured setting" in decision["rationale"]
    assert "Do not keep pushing" in decision["next_instructions"]
    assert "do not lower density further" in decision["next_instructions"]


def test_flux_reviewer_switches_when_lower_pisa_density_is_already_worse(tmp_path):
    history = [
        _checker_history(
            "pisa",
            speedup=0.8525,
            speedup_valid=False,
            qualitative_pass=True,
            quality_label="minor_regression",
            psnr=14.73,
            params={"density": 0.25, "block_size": 128, "dense_initial_steps": 0},
        ),
        _checker_history(
            "pisa",
            speedup=0.8799,
            speedup_valid=False,
            qualitative_pass=True,
            quality_label="minor_regression",
            psnr=13.56,
            params={"density": 0.125, "block_size": 128, "dense_initial_steps": 0},
        ),
    ]
    current = _checker_history(
        "pisa",
        speedup=1.005,
        speedup_valid=False,
        qualitative_pass=True,
        quality_label="similar",
        psnr=17.74,
        params={"density": 0.5, "block_size": 128, "dense_initial_steps": 0},
    )
    current.update(
        {
            "recommendation": "NEEDS_FIX",
            "command": {"exit_code": 0},
            "git_diff_unchanged": True,
        }
    )

    decision = _run_after_checker(
        tmp_path,
        diff_text="diff --git a/x b/x\n",
        executor={
            "status": "completed",
            "summary": "changed implementation",
            "command": {"exit_code": 0},
        },
        checker=current,
        history_checkers=history,
    )

    assert decision["decision"] == "NEEDS_FIX"
    assert decision["next_action_hint"]["variant"] == "spargeattn"
    assert "density direction is already bracketed" in decision["rationale"]
    assert "Do not set PISA density" in decision["next_instructions"]


def test_flux_reviewer_switches_after_bracketed_parameter_frontier(tmp_path):
    history = [
        _checker_history("pisa", speedup=0.97, speedup_valid=False, qualitative_pass=True),
        _checker_history("pisa", speedup=0.99, speedup_valid=False, qualitative_pass=True),
        _checker_history(
            "pisa",
            speedup=1.01,
            speedup_valid=True,
            qualitative_pass=False,
            quality_label="major_regression",
        ),
    ]
    current = _checker_history(
        "pisa",
        speedup=0.98,
        speedup_valid=False,
        qualitative_pass=True,
    )
    current.update(
        {
            "recommendation": "NEEDS_FIX",
            "command": {"exit_code": 0},
            "git_diff_unchanged": True,
        }
    )

    decision = _run_after_checker(
        tmp_path,
        diff_text="diff --git a/x b/x\n",
        executor={
            "status": "completed",
            "summary": "changed implementation",
            "command": {"exit_code": 0},
        },
        checker=current,
        history_checkers=history,
        action={
            "model_context": {
                "selected_group": "sparse_attention",
                "selected_variant": "pisa",
                "optimization_strategy": {
                    "available_skill_groups": {
                        "sparse_attention": ["pisa", "spargeattn"],
                        "cache": ["periodic_reuse"],
                        "quantization": ["selective_torchao_nvfp4"],
                    }
                },
            },
            "optimization_strategy": {
                "primary_skill": {"group": "sparse_attention", "variant": "pisa"}
            },
        },
    )

    assert decision["decision"] == "NEEDS_FIX"
    assert decision["next_action_hint"]["variant"] == "spargeattn"
    assert "parameter frontier is bracketed" in decision["rationale"]


def test_flux_reviewer_keeps_same_candidate_for_abnormal_reject(tmp_path):
    decision = _run_after_checker(
        tmp_path,
        diff_text="diff --git a/x b/x\n",
        executor={
            "status": "completed",
            "summary": "changed cache implementation",
            "command": {"exit_code": 0},
        },
        checker={
            "benchmark_valid": True,
            "implementation_valid": False,
            "git_diff_unchanged": True,
            "recommendation": "REJECT",
            "rationale": (
                "cache measured speedup is implausibly above skipped-step theoretical speedup. "
                "Implementation evidence failed integrity checks."
            ),
            "speed": {"delta_reuse": {"aggregate_dit_speedup": 11.56}},
            "quality": {"delta_reuse": {"mean_psnr": 27.55}},
            "qualitative": {
                "status": "completed",
                "required": True,
                "overall": {
                    "qualitative_pass": True,
                    "quality_label": "similar",
                    "summary": "Candidate image is visually comparable to baseline.",
                },
            },
            "command": {"exit_code": 0},
        },
        action={
            "model_context": {
                "selected_group": "cache",
                "selected_variant": "delta_reuse",
                "optimization_strategy": {
                    "available_skill_groups": {
                        "cache": ["delta_reuse", "ema_reuse"],
                        "quantization": ["selective_torchao_nvfp4"],
                        "sparse_attention": ["pisa"],
                    }
                },
            },
            "optimization_strategy": {
                "primary_skill": {"group": "cache", "variant": "delta_reuse"}
            },
        },
    )

    assert decision["decision"] == "NEEDS_FIX"
    assert decision["next_action_hint"]["group"] == "cache"
    assert decision["next_action_hint"]["variant"] == "delta_reuse"
    assert "same-candidate" in decision["next_action_hint"]["reason"]
    assert "Do not switch to a different skill" in decision["next_instructions"]


def test_flux_reviewer_classifies_oom_before_missing_visual_review(tmp_path):
    decision = _run_after_checker(
        tmp_path,
        diff_text="diff --git a/x b/x\n",
        executor={
            "status": "completed",
            "summary": "changed implementation",
            "command": {"exit_code": 0},
        },
        checker={
            "benchmark_valid": False,
            "implementation_valid": False,
            "git_diff_unchanged": True,
            "recommendation": "NEEDS_RETEST",
            "rationale": (
                "Benchmark command exited with code 1. metrics.json was not found. "
                "torch.OutOfMemoryError: CUDA out of memory."
            ),
            "qualitative": {
                "status": "skipped",
                "required": True,
                "overall": {
                    "qualitative_pass": None,
                    "quality_label": "inconclusive",
                    "summary": "No image rows were available in metrics.json.",
                },
            },
            "command": {"exit_code": 0},
            "artifacts": {"benchmark_exit_code": 1},
        },
    )

    assert decision["decision"] == "NEEDS_FIX"
    assert "OOM" in decision["next_action_hint"]["reason"]
    assert "memory cleanup" in decision["next_instructions"]
    assert "qualitative review must complete" not in decision["next_action_hint"]["reason"]


def test_flux_reviewer_records_pisa_block_size_resource_blocker(tmp_path):
    decision = _run_after_checker(
        tmp_path,
        diff_text="diff --git a/x b/x\n",
        executor={
            "status": "completed",
            "summary": "changed PISA block size",
            "command": {"exit_code": 0},
            "artifacts": {
                "parameter_settings": {
                    "sparse_attention": {
                        "variant": "pisa",
                        "apply_to": "single",
                        "density": 0.5,
                        "block_size": 256,
                        "dense_initial_steps": 0,
                    }
                }
            },
        },
        checker={
            "benchmark_valid": False,
            "implementation_valid": False,
            "git_diff_unchanged": True,
            "recommendation": "NEEDS_RETEST",
            "rationale": (
                "Benchmark command exited with code 1. "
                "triton.runtime.errors.OutOfResources: out of resource: shared memory."
            ),
            "qualitative": {
                "status": "skipped",
                "required": True,
                "overall": {
                    "qualitative_pass": None,
                    "quality_label": "inconclusive",
                    "summary": "No image rows were available in metrics.json.",
                },
            },
            "command": {"exit_code": 0},
            "artifacts": {"benchmark_exit_code": 1},
        },
    )

    assert decision["decision"] == "NEEDS_FIX"
    assert "block_size=256" in decision["next_action_hint"]["reason"]
    assert "Restore the last runnable PISA block_size" in decision["next_instructions"]
    assert "Do not retry the same PISA block_size" in decision["next_instructions"]


def test_flux_reviewer_next_action_uses_previous_hint_and_warmup(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "efficient_skill").mkdir()
    (workdir / "efficient_skill" / "SKILL_INDEX.md").write_text(
        "\n".join(
            [
                "## Sparse Attention",
                "- `pisa`: PISA",
                "- `spargeattn`: SpargeAttn",
                "## Cache",
                "- `periodic_reuse`: periodic",
                "## Quantization",
                "- `selective_torchao_nvfp4`: nvfp4",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    previous = run_dir / "previous_decision.json"
    action_path = run_dir / "reviewer_next_action.json"
    experiment_path = run_dir / "experiment.json"
    experiment_path.write_text(
        json.dumps(
            {
                "name": "flux",
                "goal": "optimize sparse attention",
                "metadata": {"group": "sparse_attention", "variant": "pisa"},
            }
        ),
        encoding="utf-8",
    )
    previous.write_text(
        json.dumps(
            {
                "decision": "NEEDS_FIX",
                "rationale": "continue",
                "next_action_hint": {
                    "group": "sparse_attention",
                    "variant": "spargeattn",
                    "reason": "pisa passed; continue exploring",
                },
            }
        ),
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "EAL_REVIEWER_CONTEXT": "before_executor",
        "EAL_WORKDIR": str(workdir),
        "EAL_RUN_DIR": str(run_dir),
        "EAL_ITERATION": "2",
        "EAL_EXPERIMENT_JSON": str(experiment_path),
        "EAL_REVIEWER_NEXT_ACTION": str(action_path),
        "EAL_PREVIOUS_REVIEWER_DECISION": str(previous),
        "EAL_PREVIOUS_DECISION": "NEEDS_FIX",
    }
    completed = subprocess.run(
        [sys.executable, str(REVIEWER)],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    action = json.loads(action_path.read_text(encoding="utf-8"))
    assert action["model_context"]["selected_variant"] == "spargeattn"
    assert "benchmark/flux_schnell_benchmark.py" in action["checker_commands"][0]
    assert "flux_schnell_sparse_attention_benchmark.py" not in action["checker_commands"][0]
    assert "--variants" not in action["checker_commands"][0]
    assert "--apply-to" not in action["checker_commands"][0]
    assert "--warmup-runs 1" in action["checker_commands"][0]
    assert "--width 1024 --height 1024 --steps 4" in action["checker_commands"][0]
    assert "--guidance-scale 0.0 --max-sequence-length 256" in action["checker_commands"][0]
    assert "--baseline-artifact-name eal_flux_schnell_official_baseline" in action["checker_commands"][0]
    assert "--no-warmup" not in action["checker_commands"][0]
    assert "Restricted edit surfaces" in action["instructions"]
    assert "Do not modify a restricted surface unless" in action["instructions"]
    assert "Use a run-level fixed baseline" in action["instructions"]
    assert "Parameter search is required" in action["instructions"]
    assert "Final delivery target is one composed implementation stack" in action["instructions"]

    policy = action["source_edit_policy"]
    assert policy["restricted_edit_surface"] == ["efficient_skill/", "benchmark/"]
    assert policy["allowed_restricted_reasons"] == [
        "missing_skill_capability_or_interface",
        "reusable_skill_behavior",
        "benchmark_validity",
    ]
    assert policy["current_authorization"]["efficient_skill"] is False
    assert policy["current_authorization"]["benchmark"] is False
    assert policy["current_authorization"]["reasons"] == []
    assert action["baseline_policy"]["scope"] == "run_level_fixed_baseline"
    assert action["baseline_policy"]["artifact_name"] == "eal_flux_schnell_official_baseline"
    assert action["baseline_policy"]["fixed_config"]["width"] == 1024
    assert action["baseline_policy"]["fixed_config"]["height"] == 1024
    assert action["baseline_policy"]["fixed_config"]["steps"] == 4
    assert action["baseline_policy"]["fixed_config"]["guidance_scale"] == 0.0
    assert action["baseline_policy"]["fixed_config"]["max_sequence_length"] == 256
    assert "FLUX.1-schnell model card" in action["model_context"]["official_model_config"]["source"]
    assert "missing" in " ".join(action["baseline_policy"]["rerun_baseline_only_if"])
    assert all(
        not path.startswith(("efficient_skill/", "benchmark/"))
        for path in action["default_edit_files"]
    )
    assert any(path.startswith("efficient_skill/") for path in action["restricted_edit_files"])
    assert any(path.startswith("benchmark/") for path in action["restricted_edit_files"])
    assert all(
        not path.startswith("benchmark/") for path in action["files_authorized_to_edit"]
    )
    assert all(
        not path.startswith("efficient_skill/") for path in action["files_authorized_to_edit"]
    )
    assert action["parameter_search_policy"]["mode"] == "directed_tradeoff_search"
    assert "pisa_density" in action["parameter_search_policy"]["search_dimensions"]["sparse_attention"]
    assert "compile scope: whole DiT, hot blocks, FFN/Linear path, attention path" in action["parameter_search_policy"]["search_dimensions"]["compile"]
    assert action["final_delivery_policy"]["deliverable"] == "one composed Flux optimization stack"
    assert any(
        role["role"] == "compile"
        for role in action["final_delivery_policy"]["target_stack_roles"]
    )


def test_flux_reviewer_next_action_switches_after_blocked_pisa_block_size(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "efficient_skill").mkdir()
    (workdir / "efficient_skill" / "SKILL_INDEX.md").write_text(
        "\n".join(
            [
                "## Sparse Attention",
                "- `pisa`: PISA",
                "- `spargeattn`: SpargeAttn",
                "## Cache",
                "- `periodic_reuse`: periodic",
                "## Quantization",
                "- `selective_torchao_nvfp4`: nvfp4",
            ]
        ),
        encoding="utf-8",
    )
    metrics_dir = workdir / "benchmark" / "artifacts" / "eal_checker_sparse_attention_pisa_iter003"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(
        json.dumps(
            {
                "summary": {
                    "variants": {
                        "pisa": {
                            "aggregate_dit_speedup": 1.0048,
                            "parameter_settings": {
                                "apply_to": "single",
                                "density": 0.5,
                                "block_size": 128,
                                "dense_initial_steps": 0,
                            },
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    blocked_iter = workdir / ".eal" / "runs" / "previous" / "iter-003"
    blocked_iter.mkdir(parents=True)
    (blocked_iter / "executor_result.json").write_text(
        json.dumps(
            {
                "artifacts": {
                    "final_stack": {
                        "blocked_parameter_settings": [
                            {
                                "skill_id": "pisa",
                                "role": "attention",
                                "parameters": {
                                    "apply_to": "single",
                                    "density": 0.5,
                                    "block_size": 256,
                                    "dense_initial_steps": 0,
                                },
                                "blocker": "PISA precompile failed with Triton shared-memory OutOfResources.",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    previous = run_dir / "previous_decision.json"
    action_path = run_dir / "reviewer_next_action.json"
    experiment_path = run_dir / "experiment.json"
    experiment_path.write_text(
        json.dumps({"name": "flux", "goal": "optimize sparse attention"}),
        encoding="utf-8",
    )
    previous.write_text(
        json.dumps(
            {
                "decision": "NEEDS_FIX",
                "next_action_hint": {
                    "group": "sparse_attention",
                    "variant": "pisa",
                    "reason": "quality passed but speed below target; continue directed parameter search",
                },
                "next_instructions": "For PISA, do not retry blocked parameters.",
            }
        ),
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "EAL_REVIEWER_CONTEXT": "before_executor",
        "EAL_WORKDIR": str(workdir),
        "EAL_RUN_DIR": str(run_dir),
        "EAL_ITERATION": "4",
        "EAL_EXPERIMENT_JSON": str(experiment_path),
        "EAL_REVIEWER_NEXT_ACTION": str(action_path),
        "EAL_PREVIOUS_REVIEWER_DECISION": str(previous),
        "EAL_PREVIOUS_DECISION": "NEEDS_FIX",
    }
    completed = subprocess.run(
        [sys.executable, str(REVIEWER)],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    action = json.loads(action_path.read_text(encoding="utf-8"))
    assert action["model_context"]["selected_variant"] == "spargeattn"
    assert "--variants spargeattn" not in action["checker_commands"][0]
    assert "--variants" not in action["checker_commands"][0]
    assert "block_size=256" in action["instructions"]
    assert "Previous Reviewer instructions" in action["instructions"]


def test_flux_reviewer_routes_pure_retest_to_checker(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "efficient_skill").mkdir()
    (workdir / "efficient_skill" / "SKILL_INDEX.md").write_text(
        "\n".join(
            [
                "## Sparse Attention",
                "- `pisa`: PISA",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    previous = run_dir / "previous_decision.json"
    action_path = run_dir / "reviewer_next_action.json"
    experiment_path = run_dir / "experiment.json"
    checker_prompt = run_dir / "checker_initial_prompt.md"
    checker_readme = workdir / "README_CHECKER.md"
    experiment_path.write_text(json.dumps({"name": "flux", "goal": "optimize"}), encoding="utf-8")
    checker_prompt.write_text("checker prompt", encoding="utf-8")
    checker_readme.write_text("checker readme", encoding="utf-8")
    previous.write_text(
        json.dumps(
            {
                "decision": "NEEDS_RETEST",
                "next_action_hint": {
                    "group": "sparse_attention",
                    "variant": "pisa",
                    "reason": "qualitative review must complete",
                },
            }
        ),
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "EAL_REVIEWER_CONTEXT": "before_executor",
        "EAL_WORKDIR": str(workdir),
        "EAL_RUN_DIR": str(run_dir),
        "EAL_ITERATION": "2",
        "EAL_EXPERIMENT_JSON": str(experiment_path),
        "EAL_REVIEWER_NEXT_ACTION": str(action_path),
        "EAL_PREVIOUS_REVIEWER_DECISION": str(previous),
        "EAL_PREVIOUS_DECISION": "NEEDS_RETEST",
        "EAL_CHECKER_INITIAL_PROMPT": str(checker_prompt),
        "EAL_CHECKER_README": str(checker_readme),
    }
    completed = subprocess.run(
        [sys.executable, str(REVIEWER)],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    action = json.loads(action_path.read_text(encoding="utf-8"))
    assert action["target_role"] == "Checker Step"
    assert action["target_initial_prompt"] == str(checker_prompt)
    assert action["target_readme"] == str(checker_readme)
    assert action["checker_commands"]
    assert "Rerun Checker" in action["instructions"]
    assert "do not modify code" in action["instructions"]


def test_flux_reviewer_authorizes_restricted_edits_from_previous_reason(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "efficient_skill").mkdir()
    (workdir / "efficient_skill" / "SKILL_INDEX.md").write_text(
        "\n".join(
            [
                "## Sparse Attention",
                "- `pisa`: PISA",
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    previous = run_dir / "previous_decision.json"
    action_path = run_dir / "reviewer_next_action.json"
    experiment_path = run_dir / "experiment.json"
    experiment_path.write_text(
        json.dumps({"name": "flux", "goal": "optimize sparse attention"}),
        encoding="utf-8",
    )
    previous.write_text(
        json.dumps(
            {
                "decision": "NEEDS_FIX",
                "rationale": "checker failure",
                "next_action_hint": {
                    "group": "sparse_attention",
                    "variant": "pisa",
                    "reason": "implementation did not satisfy checker after benchmark invalid artifacts",
                },
            }
        ),
        encoding="utf-8",
    )

    env = {
        **os.environ,
        "EAL_REVIEWER_CONTEXT": "before_executor",
        "EAL_WORKDIR": str(workdir),
        "EAL_RUN_DIR": str(run_dir),
        "EAL_ITERATION": "3",
        "EAL_EXPERIMENT_JSON": str(experiment_path),
        "EAL_REVIEWER_NEXT_ACTION": str(action_path),
        "EAL_PREVIOUS_REVIEWER_DECISION": str(previous),
        "EAL_PREVIOUS_DECISION": "NEEDS_FIX",
    }
    completed = subprocess.run(
        [sys.executable, str(REVIEWER)],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    action = json.loads(action_path.read_text(encoding="utf-8"))
    authorization = action["source_edit_policy"]["current_authorization"]
    assert authorization["efficient_skill"] is True
    assert authorization["benchmark"] is True
    assert "reusable_skill_behavior" in authorization["reasons"]
    assert "benchmark_validity" in authorization["reasons"]
    assert any(path.startswith("efficient_skill/") for path in action["files_authorized_to_edit"])
    assert any(path.startswith("benchmark/") for path in action["files_authorized_to_edit"])


def test_flux_reviewer_requires_final_delivery_stack_before_accept(tmp_path):
    decision = _run_after_checker(
        tmp_path,
        diff_text="diff --git a/x b/x\n",
        executor={
            "status": "completed",
            "summary": "changed implementation",
            "command": {"exit_code": 0},
        },
        checker={
            "benchmark_valid": True,
            "implementation_valid": True,
            "git_diff_unchanged": True,
            "recommendation": "ACCEPT",
            "speed": {"pisa": {"aggregate_dit_speedup": 1.2}},
            "quality": {"pisa": {"mean_psnr": 35.0}},
            "qualitative": {
                "status": "completed",
                "required": True,
                "overall": {
                    "qualitative_pass": True,
                    "quality_label": "pass",
                    "summary": "Candidate image is visually comparable to baseline.",
                },
            },
            "command": {"exit_code": 0},
        },
        action={
            "model_context": {
                "selected_group": "sparse_attention",
                "selected_variant": "pisa",
                "optimization_strategy": {
                    "available_skill_groups": {
                        "sparse_attention": ["pisa"],
                        "cache": [],
                        "quantization": [],
                        "compile": ["torch_compile"],
                    }
                },
            },
            "optimization_strategy": {
                "primary_skill": {"group": "sparse_attention", "variant": "pisa"}
            },
        },
    )

    assert decision["decision"] == "NEEDS_FIX"
    assert "no final composed delivery stack" in decision["rationale"]
    assert "final composed Flux delivery stack" in decision["next_instructions"]
    assert "torch_compile" in decision["next_action_hint"]["reason"]


def _run_after_checker(
    tmp_path: Path,
    *,
    diff_text: str,
    executor: dict,
    checker: dict,
    action: dict | None = None,
    history_checkers: list[dict] | None = None,
) -> dict:
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    for index, historical_checker in enumerate(history_checkers or [], start=1):
        history_path = workdir / ".eal" / "runs" / "history" / f"iter-{index:03d}"
        history_path.mkdir(parents=True, exist_ok=True)
        (history_path / "checker_result.json").write_text(
            json.dumps(historical_checker),
            encoding="utf-8",
        )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checker_path = run_dir / "checker_result.json"
    executor_path = run_dir / "executor_result.json"
    action_path = run_dir / "reviewer_next_action.json"
    diff_path = run_dir / "git_diff.patch"
    decision_path = run_dir / "reviewer_decision.json"
    experiment_path = run_dir / "experiment.json"

    checker_path.write_text(json.dumps(checker), encoding="utf-8")
    executor_path.write_text(json.dumps(executor), encoding="utf-8")
    diff_path.write_text(diff_text, encoding="utf-8")
    experiment_path.write_text(json.dumps({"name": "flux", "goal": "optimize"}), encoding="utf-8")
    if action is None:
        action = {
            "model_context": {
                "selected_group": "sparse_attention",
                "selected_variant": "pisa",
                "optimization_strategy": {
                    "available_skill_groups": {
                        "cache": ["periodic_reuse"],
                        "quantization": ["selective_torchao_nvfp4"],
                        "sparse_attention": ["pisa", "spargeattn"],
                    }
                },
            },
            "optimization_strategy": {
                "primary_skill": {"group": "sparse_attention", "variant": "pisa"}
            },
        }
    action_path.write_text(json.dumps(action), encoding="utf-8")

    env = {
        **os.environ,
        "EAL_REVIEWER_CONTEXT": "after_checker",
        "EAL_WORKDIR": str(workdir),
        "EAL_RUN_DIR": str(run_dir),
        "EAL_ITERATION": "1",
        "EAL_CHECKER_RESULT": str(checker_path),
        "EAL_EXECUTOR_RESULT": str(executor_path),
        "EAL_REVIEWER_NEXT_ACTION": str(action_path),
        "EAL_REVIEWER_DECISION": str(decision_path),
        "EAL_EXPERIMENT_JSON": str(experiment_path),
        "EAL_GIT_DIFF": str(diff_path),
    }
    completed = subprocess.run(
        [sys.executable, str(REVIEWER)],
        cwd=workdir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(decision_path.read_text(encoding="utf-8"))


def _checker_history(
    variant: str,
    *,
    speedup: float,
    speedup_valid: bool,
    qualitative_pass: bool,
    quality_label: str = "minor_regression",
    psnr: float = 25.0,
    params: dict | None = None,
) -> dict:
    data = {
        "benchmark_valid": True,
        "implementation_valid": False,
        "speed": {
            variant: {
                "aggregate_dit_speedup": speedup,
                "speedup_valid": speedup_valid,
            }
        },
        "quality": {variant: {"mean_psnr": psnr, "quality_valid": True}},
        "qualitative": {
            "status": "completed",
            "required": True,
            "overall": {
                "qualitative_pass": qualitative_pass,
                "quality_label": quality_label,
                "summary": "historical visual result",
            },
        },
    }
    if params:
        data["parameter_settings"] = {variant: params}
    return data
