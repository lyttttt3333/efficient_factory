from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


RECOMMENDATION_MAP = {
    "accept": "ACCEPT",
    "needs_review": "NEEDS_FIX",
    "iterate": "NEEDS_FIX",
    "reject": "REJECT",
    "rerun": "NEEDS_RETEST",
}

def main() -> int:
    workdir = Path(os.environ["EAL_WORKDIR"]).resolve()
    run_dir = Path(os.environ["EAL_RUN_DIR"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    action = _read_json(Path(os.environ["EAL_REVIEWER_NEXT_ACTION"]))
    command = _checker_command(action)
    stdout_log = run_dir / "checker_benchmark.stdout.txt"
    stderr_log = run_dir / "checker_benchmark.stderr.txt"

    started = time.monotonic()
    gpu_preflight = _gpu_preflight_check()
    if gpu_preflight.get("status") == "blocked":
        reason = _gpu_preflight_message(gpu_preflight)
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text(reason + "\n", encoding="utf-8")
        result = _scheduler_checker_result(
            raw_result={},
            visual_result={},
            action=action,
            metrics_path=_metrics_path(workdir, action),
            converted_path=run_dir / "checker_result_from_metrics.json",
            command=command,
            exit_code=0,
            duration=round(time.monotonic() - started, 6),
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            conversion_error=reason,
            gpu_preflight=gpu_preflight,
        )
        result["rationale"] = reason
        result["recommendation"] = "NEEDS_RETEST"
        result["benchmark_valid"] = False
        result["implementation_valid"] = False
        result["verdict"] = "BENCHMARK_INVALID"
        result["artifacts"]["benchmark_skipped_reason"] = reason
        _write_json(Path(os.environ["EAL_CHECKER_RESULT"]), result)
        return 0

    exit_code, gpu_runtime = _run_streaming(command, workdir, stdout_log, stderr_log)
    duration = round(time.monotonic() - started, 6)

    metrics_path = _metrics_path(workdir, action)
    converted_path = run_dir / "checker_result_from_metrics.json"
    raw_result: dict[str, Any] = {}
    conversion_error = ""
    if metrics_path.exists():
        raw_result, conversion_error = _convert_metrics(
            workdir,
            metrics_path,
            converted_path,
            allow_no_warmup="--no-warmup" in command.split(),
        )
    visual_result = _run_visual_quality(workdir, run_dir, metrics_path, action)

    result = _scheduler_checker_result(
        raw_result=raw_result,
        visual_result=visual_result,
        action=action,
        metrics_path=metrics_path,
        converted_path=converted_path,
        command=command,
        exit_code=exit_code,
        duration=duration,
        stdout_log=stdout_log,
        stderr_log=stderr_log,
        conversion_error=conversion_error,
        gpu_preflight=gpu_preflight,
        gpu_runtime=gpu_runtime,
    )
    _write_json(Path(os.environ["EAL_CHECKER_RESULT"]), result)
    return 0


def _checker_command(action: dict[str, Any]) -> str:
    commands = action.get("checker_commands")
    if isinstance(commands, list) and commands and isinstance(commands[0], str):
        return commands[0]
    plan = action.get("plan")
    if isinstance(plan, dict):
        nested = plan.get("benchmark_commands")
        if isinstance(nested, list) and nested and isinstance(nested[0], str):
            return nested[0]
    raise SystemExit("Reviewer next action did not contain checker_commands.")


def _metrics_path(workdir: Path, action: dict[str, Any]) -> Path:
    context = action.get("model_context")
    if isinstance(context, dict) and isinstance(context.get("artifact_name"), str):
        return workdir / "benchmark" / "artifacts" / context["artifact_name"] / "metrics.json"
    return workdir / "benchmark" / "artifacts" / "metrics.json"


def _run_streaming(command: str, cwd: Path, stdout_log: Path, stderr_log: Path) -> tuple[int, dict[str, Any]]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )
    gpu_monitor = _start_gpu_runtime_monitor(process.pid)
    threads = [
        threading.Thread(target=_copy_stream, args=(process.stdout, stdout_log), daemon=True),
        threading.Thread(target=_copy_stream, args=(process.stderr, stderr_log), daemon=True),
    ]
    for thread in threads:
        thread.start()
    exit_code = process.wait()
    gpu_runtime = _stop_gpu_runtime_monitor(gpu_monitor)
    for thread in threads:
        thread.join()
    return exit_code, gpu_runtime


def _convert_metrics(
    workdir: Path,
    metrics_path: Path,
    converted_path: Path,
    *,
    allow_no_warmup: bool,
) -> tuple[dict[str, Any], str]:
    command = [
        "conda",
        "run",
        "-n",
        "auto_deploy_flux_eff",
        "python",
        "-B",
        "agent_loop/metrics_to_checker_result.py",
        "--metrics",
        str(metrics_path),
        "--output",
        str(converted_path),
        "--repo-root",
        str(workdir),
        "--diff-report",
        "agent_loop_state/executor_diff.json",
        "--static-audit-status",
        "pass",
    ]
    if allow_no_warmup:
        command.append("--allow-no-warmup")
    completed = subprocess.run(
        command,
        cwd=workdir,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return {}, completed.stderr.strip() or completed.stdout.strip()
    if not converted_path.exists():
        return {}, "metrics_to_checker_result.py did not write output."
    return _read_json(converted_path), ""


def _scheduler_checker_result(
    *,
    raw_result: dict[str, Any],
    visual_result: dict[str, Any],
    action: dict[str, Any] | None = None,
    metrics_path: Path,
    converted_path: Path,
    command: str,
    exit_code: int,
    duration: float,
    stdout_log: Path,
    stderr_log: Path,
    conversion_error: str,
    gpu_preflight: dict[str, Any] | None = None,
    gpu_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_recommendation = str(raw_result.get("recommendation", "rerun")).lower()
    benchmark_valid = bool(raw_result.get("benchmark_valid", False)) and exit_code == 0
    implementation_valid = benchmark_valid and raw_recommendation == "accept"
    recommendation = RECOMMENDATION_MAP.get(raw_recommendation, "NEEDS_RETEST")
    if exit_code != 0 or conversion_error or not metrics_path.exists():
        benchmark_valid = False
        implementation_valid = False
        recommendation = "NEEDS_RETEST"
    if gpu_runtime and gpu_runtime.get("status") == "blocked":
        benchmark_valid = False
        implementation_valid = False
        recommendation = "NEEDS_RETEST"

    config_mismatch = _benchmark_config_mismatch(raw_result, action or {})
    if benchmark_valid and config_mismatch:
        benchmark_valid = False
        implementation_valid = False
        recommendation = "NEEDS_RETEST"

    qualitative = _qualitative_summary(visual_result)
    visual_enabled = bool(visual_result)
    visual_required = bool(visual_result.get("required", False)) if visual_enabled else False
    visual_status = str(visual_result.get("status", "")) if visual_enabled else ""
    visual_overall = visual_result.get("overall", {}) if isinstance(visual_result.get("overall"), dict) else {}
    visual_pass = visual_overall.get("qualitative_pass")
    if visual_enabled and visual_required and visual_status != "completed":
        benchmark_valid = False
        implementation_valid = False
        recommendation = "NEEDS_RETEST"
    elif visual_enabled and visual_status == "completed" and visual_pass is False:
        implementation_valid = False
        if recommendation == "ACCEPT":
            recommendation = "NEEDS_FIX"

    failed = _nested_list(raw_result, "integrity_checks", "failed")
    issues = raw_result.get("implementation_issues", [])
    notes = raw_result.get("next_reviewer_notes", [])
    abnormal_reason = _abnormal_raw_evidence_reason(raw_result, failed, issues, notes)
    if benchmark_valid and abnormal_reason:
        implementation_valid = False
        recommendation = "REJECT"

    rationale_parts = []
    if exit_code != 0:
        rationale_parts.append(f"Benchmark command exited with code {exit_code}.")
    if not metrics_path.exists():
        rationale_parts.append(f"metrics.json was not found at {metrics_path}.")
    if conversion_error:
        rationale_parts.append(f"Could not convert metrics: {conversion_error}")
    if gpu_runtime and gpu_runtime.get("status") == "blocked":
        rationale_parts.append(_gpu_runtime_message(gpu_runtime))
    if config_mismatch:
        rationale_parts.append(config_mismatch)
    if visual_enabled:
        visual_summary = str(visual_overall.get("summary", "")).strip()
        if visual_status != "completed":
            rationale_parts.append(f"Visual qualitative check did not complete: {visual_summary or visual_status}.")
        elif visual_pass is False:
            rationale_parts.append(f"Visual qualitative check failed: {visual_summary}")
        elif visual_summary:
            rationale_parts.append(f"Visual qualitative check passed: {visual_summary}")
    rationale_parts.extend(str(item) for item in failed[:8])
    if isinstance(issues, list):
        rationale_parts.extend(str(item) for item in issues[:8])
    if isinstance(notes, list):
        rationale_parts.extend(str(item) for item in notes[:4])
    if abnormal_reason and abnormal_reason not in " ".join(rationale_parts):
        rationale_parts.append(abnormal_reason)
    rationale = " ".join(rationale_parts).strip() or "Benchmark completed and metrics were converted."

    return {
        "iteration": int(os.environ.get("EAL_ITERATION", "1")),
        "status": "completed",
        "benchmark_valid": benchmark_valid,
        "implementation_valid": implementation_valid,
        "verdict": _verdict(benchmark_valid, implementation_valid),
        "recommendation": recommendation,
        "rationale": rationale,
        "speed": _speed_summary(raw_result, action or {}),
        "quality": _quality_summary(raw_result, action or {}),
        "qualitative": qualitative,
        "artifacts": {
            "benchmark_stdout": str(stdout_log),
            "benchmark_stderr": str(stderr_log),
            "metrics": str(metrics_path),
            "converted_checker_result": str(converted_path),
            "benchmark_command": command,
            "benchmark_exit_code": exit_code,
            "benchmark_duration_seconds": duration,
            **({"parameter_settings": _executor_parameter_settings()} if _executor_parameter_settings() else {}),
            **({"gpu_preflight": gpu_preflight} if gpu_preflight else {}),
            **({"gpu_runtime_monitor": gpu_runtime} if gpu_runtime else {}),
            **({"visual_quality": str(Path(visual_result["output_path"]))} if visual_result.get("output_path") else {}),
        },
    }


def _benchmark_config_mismatch(raw_result: dict[str, Any], action: dict[str, Any]) -> str:
    policy = action.get("baseline_policy")
    if not isinstance(policy, dict):
        return ""
    fixed = policy.get("fixed_config")
    if not isinstance(fixed, dict):
        return ""
    summary = raw_result.get("summary")
    if not isinstance(summary, dict):
        return ""
    benchmark_config = summary.get("benchmark_config")
    if not isinstance(benchmark_config, dict):
        return ""

    mismatches = []
    for key in ("width", "height", "steps"):
        if key not in fixed:
            continue
        expected = fixed.get(key)
        actual = benchmark_config.get(key)
        if actual != expected:
            mismatches.append(f"{key} expected {expected!r} got {actual!r}")

    expected_baseline = policy.get("artifact_name")
    actual_baseline = benchmark_config.get("baseline_artifact_name")
    if actual_baseline and expected_baseline and actual_baseline != expected_baseline:
        mismatches.append(
            f"baseline_artifact_name expected {expected_baseline!r} got {actual_baseline!r}"
        )

    if not mismatches:
        return ""
    return (
        "Benchmark config does not match the fixed official model hyperparameters "
        "from reviewer_next_action: "
        + "; ".join(mismatches)
    )


def _abnormal_raw_evidence_reason(
    raw_result: dict[str, Any],
    failed: list[Any],
    issues: Any,
    notes: Any,
) -> str:
    text = " ".join(
        part
        for part in (
            _safe_json(raw_result.get("speed", {})),
            _safe_json(raw_result.get("quality", {})),
            " ".join(str(item) for item in failed),
            " ".join(str(item) for item in issues) if isinstance(issues, list) else str(issues or ""),
            " ".join(str(item) for item in notes) if isinstance(notes, list) else str(notes or ""),
        )
        if part
    ).lower()
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
        return "abnormal benchmark or implementation evidence requires rejection and same-candidate repair"
    return ""


def _safe_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _gpu_preflight_check() -> dict[str, Any]:
    if os.environ.get("EAL_SKIP_GPU_PREFLIGHT", "").strip().lower() in {"1", "true", "yes"}:
        return {"status": "skipped", "reason": "disabled_by_EAL_SKIP_GPU_PREFLIGHT"}

    threshold_mib = int(os.environ.get("EAL_GPU_PREFLIGHT_MAX_EXTERNAL_MIB", "1024"))
    visible_bus_ids = _visible_gpu_bus_ids()
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_bus_id,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "unavailable",
            "reason": f"nvidia-smi preflight unavailable: {exc}",
            "threshold_mib": threshold_mib,
        }

    if completed.returncode != 0:
        return {
            "status": "unavailable",
            "reason": (completed.stderr or completed.stdout or "nvidia-smi failed").strip(),
            "threshold_mib": threshold_mib,
        }

    processes = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        gpu_bus_id, pid_text, process_name, memory_text = parts
        if visible_bus_ids is not None and gpu_bus_id not in visible_bus_ids:
            continue
        try:
            pid = int(pid_text)
            memory_mib = int(memory_text)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        if memory_mib >= threshold_mib:
            processes.append(
                {
                    "gpu_bus_id": gpu_bus_id,
                    "pid": pid,
                    "process_name": process_name,
                    "used_gpu_memory_mib": memory_mib,
                }
            )

    return {
        "status": "blocked" if processes else "clear",
        "threshold_mib": threshold_mib,
        "external_compute_processes": processes,
    }


def _gpu_preflight_message(gpu_preflight: dict[str, Any]) -> str:
    processes = gpu_preflight.get("external_compute_processes")
    if not isinstance(processes, list):
        processes = []
    details = ", ".join(
        f"pid={item.get('pid')} name={item.get('process_name')} mem={item.get('used_gpu_memory_mib')}MiB"
        for item in processes
        if isinstance(item, dict)
    )
    threshold = gpu_preflight.get("threshold_mib")
    return (
        "GPU preflight blocked benchmark: external compute process usage exceeded "
        f"{threshold} MiB. {details}"
    ).strip()


def _start_gpu_runtime_monitor(root_pid: int) -> dict[str, Any]:
    if os.environ.get("EAL_SKIP_GPU_PREFLIGHT", "").strip().lower() in {"1", "true", "yes"}:
        return {
            "status": "skipped",
            "reason": "disabled_by_EAL_SKIP_GPU_PREFLIGHT",
            "root_pid": root_pid,
        }
    threshold_mib = int(os.environ.get("EAL_GPU_PREFLIGHT_MAX_EXTERNAL_MIB", "1024"))
    interval = float(os.environ.get("EAL_GPU_MONITOR_INTERVAL_SECONDS", "1.0"))
    max_samples = int(os.environ.get("EAL_GPU_MONITOR_MAX_SAMPLES_RECORDED", "12"))
    stop_event = threading.Event()
    state: dict[str, Any] = {
        "status": "clear",
        "root_pid": root_pid,
        "threshold_mib": threshold_mib,
        "interval_seconds": interval,
        "samples_count": 0,
        "external_compute_processes": [],
        "samples": [],
        "_stop_event": stop_event,
    }
    thread = threading.Thread(
        target=_gpu_runtime_monitor_loop,
        args=(root_pid, threshold_mib, interval, max_samples, stop_event, state),
        daemon=True,
    )
    state["_thread"] = thread
    thread.start()
    return state


def _stop_gpu_runtime_monitor(state: dict[str, Any]) -> dict[str, Any]:
    stop_event = state.get("_stop_event")
    if hasattr(stop_event, "set"):
        stop_event.set()
    thread = state.get("_thread")
    if isinstance(thread, threading.Thread):
        thread.join(timeout=5)
    return {key: value for key, value in state.items() if not key.startswith("_")}


def _gpu_runtime_monitor_loop(
    root_pid: int,
    threshold_mib: int,
    interval: float,
    max_samples: int,
    stop_event: threading.Event,
    state: dict[str, Any],
) -> None:
    while not stop_event.is_set():
        sample = _gpu_runtime_sample(root_pid, threshold_mib)
        state["samples_count"] = int(state.get("samples_count", 0)) + 1
        samples = state.setdefault("samples", [])
        if isinstance(samples, list) and len(samples) < max_samples:
            samples.append(sample)
        external = sample.get("external_compute_processes", [])
        if external:
            state["status"] = "blocked"
            current = state.setdefault("external_compute_processes", [])
            if isinstance(current, list):
                _merge_external_processes(current, external)
        stop_event.wait(interval)


def _gpu_runtime_sample(root_pid: int, threshold_mib: int) -> dict[str, Any]:
    benchmark_pids = _process_descendants(root_pid)
    benchmark_pids.add(root_pid)
    benchmark_pids.add(os.getpid())
    processes = _query_gpu_processes()
    external = [
        item
        for item in processes
        if int(item.get("pid", -1)) not in benchmark_pids
        and int(item.get("used_gpu_memory_mib", 0)) >= threshold_mib
    ]
    return {
        "created_at_utc": _utc_now(),
        "benchmark_process_pids": sorted(benchmark_pids),
        "gpu_processes": processes,
        "external_compute_processes": external,
    }


def _query_gpu_processes() -> list[dict[str, Any]]:
    visible_bus_ids = _visible_gpu_bus_ids()
    command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_bus_id,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, text=True, capture_output=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if completed.returncode != 0:
        return []
    processes = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        gpu_bus_id, pid_text, process_name, memory_text = parts
        if visible_bus_ids is not None and gpu_bus_id not in visible_bus_ids:
            continue
        try:
            pid = int(pid_text)
            memory_mib = int(memory_text)
        except ValueError:
            continue
        processes.append(
            {
                "gpu_bus_id": gpu_bus_id,
                "pid": pid,
                "process_name": process_name,
                "used_gpu_memory_mib": memory_mib,
            }
        )
    return processes


def _visible_gpu_bus_ids() -> set[str] | None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.strip():
        return None
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if not tokens or tokens == ["-1"]:
        return set()
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,gpu_uuid,gpu_bus_id",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    by_index: dict[str, str] = {}
    by_uuid: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        index, uuid, bus_id = parts
        by_index[index] = bus_id
        by_uuid[uuid] = bus_id
    selected = {
        by_index[token] if token in by_index else by_uuid[token]
        for token in tokens
        if token in by_index or token in by_uuid
    }
    return selected if selected else None


def _process_descendants(root_pid: int) -> set[int]:
    try:
        completed = subprocess.run(
            ["ps", "-eo", "pid=,ppid="],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return set()
    if completed.returncode != 0:
        return set()

    children_by_parent: dict[int, list[int]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        children_by_parent.setdefault(ppid, []).append(pid)

    descendants: set[int] = set()
    stack = list(children_by_parent.get(root_pid, []))
    while stack:
        pid = stack.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        stack.extend(children_by_parent.get(pid, []))
    return descendants


def _merge_external_processes(current: list[Any], external: Any) -> None:
    seen = {
        (item.get("pid"), item.get("process_name"))
        for item in current
        if isinstance(item, dict)
    }
    if not isinstance(external, list):
        return
    for item in external:
        if not isinstance(item, dict):
            continue
        key = (item.get("pid"), item.get("process_name"))
        if key in seen:
            continue
        current.append(item)
        seen.add(key)


def _gpu_runtime_message(gpu_runtime: dict[str, Any]) -> str:
    processes = gpu_runtime.get("external_compute_processes")
    if not isinstance(processes, list):
        processes = []
    details = ", ".join(
        f"pid={item.get('pid')} name={item.get('process_name')} mem={item.get('used_gpu_memory_mib')}MiB"
        for item in processes
        if isinstance(item, dict)
    )
    threshold = gpu_runtime.get("threshold_mib")
    return (
        "GPU runtime monitor invalidated benchmark: external compute process usage exceeded "
        f"{threshold} MiB while benchmark was running. {details}"
    ).strip()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _run_visual_quality(
    workdir: Path,
    run_dir: Path,
    metrics_path: Path,
    action: dict[str, Any],
) -> dict[str, Any]:
    config = action.get("visual_quality")
    if not isinstance(config, dict) or not config.get("enabled", False):
        return {}

    output_path = run_dir / "checker_visual_quality.json"
    stdout_path = run_dir / "checker_visual_quality.stdout.txt"
    stderr_path = run_dir / "checker_visual_quality.stderr.txt"
    script = Path(__file__).with_name("visual_quality_from_metrics.py")
    command = [
        sys.executable,
        str(script),
        "--metrics",
        str(metrics_path),
        "--output",
        str(output_path),
        "--repo-root",
        str(workdir),
        "--max-pairs",
        str(int(config.get("max_pairs", 1))),
    ]
    model = str(config.get("model", "")).strip()
    if model:
        command.extend(["--model", model])
    if not bool(config.get("required", True)):
        command.append("--optional")
    timeout = float(config.get("timeout_seconds", os.environ.get("EAL_VISUAL_REVIEW_TIMEOUT_SECONDS", "300")))

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout or "", encoding="utf-8")
        stderr_path.write_text(exc.stderr or "", encoding="utf-8")
        result = {
            "schema_version": "auto_deploy.visual_quality.v1",
            "status": "failed",
            "required": bool(config.get("required", True)),
            "output_path": str(output_path),
            "overall": {
                "qualitative_pass": None,
                "quality_label": "inconclusive",
                "summary": f"Visual quality command timed out after {timeout:.1f}s.",
                "recommendation": "rerun",
            },
        }
        _write_json(output_path, result)
        return result

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    if output_path.exists():
        result = _read_json(output_path)
    else:
        result = {
            "schema_version": "auto_deploy.visual_quality.v1",
            "status": "failed",
            "required": bool(config.get("required", True)),
            "overall": {
                "qualitative_pass": None,
                "quality_label": "inconclusive",
                "summary": "Visual quality command did not write checker_visual_quality.json.",
                "recommendation": "rerun",
            },
        }
    result["output_path"] = str(output_path)
    result["command"] = command
    result["exit_code"] = completed.returncode
    result["duration_seconds"] = round(time.monotonic() - started, 6)
    result["stdout"] = str(stdout_path)
    result["stderr"] = str(stderr_path)
    _write_json(output_path, result)
    return result


def _qualitative_summary(visual_result: dict[str, Any]) -> dict[str, Any]:
    if not visual_result:
        return {}
    overall = visual_result.get("overall", {})
    if not isinstance(overall, dict):
        overall = {}
    return {
        "schema_version": visual_result.get("schema_version"),
        "status": visual_result.get("status"),
        "required": visual_result.get("required"),
        "overall": overall,
        "pairs": visual_result.get("pairs", []),
        "artifacts": {
            "visual_quality": visual_result.get("output_path"),
            "stdout": visual_result.get("stdout"),
            "stderr": visual_result.get("stderr"),
        },
    }


def _verdict(benchmark_valid: bool, implementation_valid: bool) -> str:
    if not benchmark_valid:
        return "BENCHMARK_INVALID"
    if implementation_valid:
        return "VALID_EFFECTIVE"
    return "VALID_INEFFECTIVE"


def _speed_summary(raw_result: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    speed: dict[str, Any] = {}
    for item in raw_result.get("variant_results", []) or []:
        if not isinstance(item, dict):
            continue
        variant = str(item.get("variant", "unknown"))
        speed[variant] = {
            "aggregate_dit_speedup": item.get("aggregate_dit_speedup"),
            "baseline_mean_dit_wall_time_s": item.get("baseline_mean_dit_wall_time_s"),
            "variant_mean_dit_wall_time_s": item.get("variant_mean_dit_wall_time_s"),
            "speedup_valid": item.get("speedup_valid"),
        }
    _add_selected_variant_alias(speed, action)
    return speed


def _quality_summary(raw_result: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    quality: dict[str, Any] = {}
    for item in raw_result.get("variant_results", []) or []:
        if not isinstance(item, dict):
            continue
        variant = str(item.get("variant", "unknown"))
        quality[variant] = {
            "mean_mse": item.get("mean_mse"),
            "mean_mae": item.get("mean_mae"),
            "mean_psnr": item.get("mean_psnr"),
            "quality_valid": item.get("quality_valid"),
        }
    _add_selected_variant_alias(quality, action)
    return quality


def _add_selected_variant_alias(values: dict[str, Any], action: dict[str, Any]) -> None:
    selected_variant = _selected_variant(action)
    if not selected_variant or selected_variant in values or len(values) != 1:
        return
    only_value = next(iter(values.values()))
    if isinstance(only_value, dict):
        values[selected_variant] = dict(only_value)


def _selected_variant(action: dict[str, Any]) -> str:
    context = action.get("model_context")
    if isinstance(context, dict):
        value = context.get("selected_variant")
        if isinstance(value, str) and value.strip():
            return value.strip()
        strategy = context.get("optimization_strategy")
        if isinstance(strategy, dict):
            primary = strategy.get("primary_skill") or strategy.get("primary_candidate")
            if isinstance(primary, dict):
                value = primary.get("variant")
                if isinstance(value, str) and value.strip():
                    return value.strip()
    strategy = action.get("optimization_strategy")
    if isinstance(strategy, dict):
        primary = strategy.get("primary_skill") or strategy.get("primary_candidate")
        if isinstance(primary, dict):
            value = primary.get("variant")
            if isinstance(value, str) and value.strip():
                return value.strip()
    value = action.get("selected_variant")
    return value.strip() if isinstance(value, str) else ""


def _executor_parameter_settings() -> dict[str, Any]:
    executor_path = Path(os.environ.get("EAL_EXECUTOR_RESULT", ""))
    if not executor_path.exists():
        return {}
    try:
        executor = _read_json(executor_path)
    except (OSError, json.JSONDecodeError):
        return {}
    artifacts = executor.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return {}
    settings = artifacts.get("parameter_settings", {})
    return settings if isinstance(settings, dict) else {}


def _nested_list(data: dict[str, Any], outer: str, inner: str) -> list[Any]:
    value = data.get(outer)
    if not isinstance(value, dict):
        return []
    nested = value.get(inner)
    return nested if isinstance(nested, list) else []


def _copy_stream(stream, path: Path) -> None:
    if stream is None:
        return
    with path.open("w", encoding="utf-8") as handle:
        for line in stream:
            handle.write(line)
            handle.flush()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
