from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from typing import Any

from .agents import (
    AgentPaths,
    CheckerStep,
    ExecutorAgent,
    ReviewerAgent,
    ensure_executor_has_diff_explanation,
)
from .git_utils import collect_diff, commit_all, ensure_git_repo, rollback_worktree
from .prompts import load_prompt
from .schemas import (
    Decision,
    ExperimentSpec,
    LabConfig,
    LoopResult,
    ReviewerDecision,
    read_json,
    write_json,
)


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        role: str,
        event: str,
        message: str,
        iteration: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        import json

        row: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "event": event,
            "message": message,
        }
        if iteration is not None:
            row["iteration"] = iteration
        if data:
            row["data"] = data
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


class Scheduler:
    def __init__(
        self,
        experiment: ExperimentSpec,
        lab: LabConfig,
        *,
        workdir: str | Path | None = None,
    ) -> None:
        self.experiment = experiment
        self.lab = lab
        self.workdir = Path(workdir or lab.workdir).resolve()
        self.executor = ExecutorAgent()
        self.checker = CheckerStep()
        self.reviewer = ReviewerAgent()

    def run(self, *, resume_run_root: str | Path | None = None) -> LoopResult:
        repo_root = ensure_git_repo(self.workdir)
        run_root = (
            Path(resume_run_root).resolve()
            if resume_run_root is not None
            else self._make_run_root(repo_root)
        )
        events = EventLog(run_root / "events.jsonl")
        experiment_json = run_root / "experiment.json"
        if resume_run_root is None or not experiment_json.exists():
            write_json(experiment_json, self.experiment.to_dict())
        role_paths = _write_initial_prompts(run_root, repo_root, self.lab)
        resume_state = _resume_state(run_root) if resume_run_root is not None else None
        if resume_state is None:
            events.record(
                role="Scheduler",
                event="run_started",
                message="Created run directory and loaded experiment.",
                data={
                    "experiment_name": self.experiment.name,
                    "workdir": str(repo_root),
                    "run_dir": str(run_root),
                },
            )
        else:
            events.record(
                role="Scheduler",
                event="run_resumed",
                message="Resumed an existing run directory.",
                data={
                    "experiment_name": self.experiment.name,
                    "workdir": str(repo_root),
                    "run_dir": str(run_root),
                    "start_iteration": resume_state["start_iteration"],
                    "previous_iteration": resume_state["previous_iteration"],
                    "previous_decision": resume_state["previous_decision"].decision.value,
                },
            )
        events.record(
            role="Scheduler",
            event="initial_prompts_written",
            message="Wrote role initial prompts that point each agent to its README.",
            data={key: str(value) for key, value in role_paths.items()},
        )

        iterations: list[dict[str, object]] = []
        previous_decision: ReviewerDecision | None = (
            resume_state["previous_decision"] if resume_state is not None else None
        )
        mode = (
            "RETEST"
            if previous_decision is not None
            and previous_decision.decision == Decision.NEEDS_RETEST
            else "EXECUTE"
        )
        if resume_state is not None and mode == "RETEST":
            latest_checker = (
                run_root
                / f"iter-{resume_state['previous_iteration']:03d}"
                / "checker_result.json"
            )
            _hold_reviewer_until_resources_clean(
                self.lab,
                events,
                int(resume_state["previous_iteration"]),
                latest_checker,
            )
        final_decision = Decision.STOP
        status = "stopped"
        start_iteration = (
            int(resume_state["start_iteration"]) if resume_state is not None else 1
        )

        iteration_numbers = (
            count(start_iteration)
            if self.lab.max_iterations == 0
            else range(start_iteration, start_iteration + self.lab.max_iterations)
        )
        for iteration in iteration_numbers:
            events.record(
                role="Scheduler",
                event="iteration_started",
                message=f"Starting iteration in {mode} mode.",
                iteration=iteration,
                data={"mode": mode},
            )
            paths = self._paths_for_iteration(
                run_root,
                experiment_json,
                iteration,
                previous_decision,
                role_paths,
            )
            paths.run_dir.mkdir(parents=True, exist_ok=True)

            events.record(
                role="Reviewer Agent",
                event="wake",
                message="Reviewer is deciding the next action from ExperimentSpec and prior feedback.",
                iteration=iteration,
                data={
                    "permission": self.lab.reviewer_permission.value,
                    "experiment_json": str(experiment_json),
                    "initial_prompt": str(paths.reviewer_initial_prompt),
                    "role_readme": str(paths.reviewer_readme),
                    "previous_decision": (
                        previous_decision.decision.value
                        if previous_decision is not None
                        else ""
                    ),
                },
            )
            if mode == "RETEST":
                default_task_target = "Checker Step"
                default_task_instructions = (
                    previous_decision.next_instructions
                    if previous_decision is not None
                    and previous_decision.next_instructions
                    else "Rerun benchmark because the previous checker result was not trusted."
                )
            else:
                default_task_target = "Executor Agent"
                default_task_instructions = (
                    previous_decision.next_instructions
                    if previous_decision is not None
                    and previous_decision.next_instructions
                    else self.experiment.instructions or self.experiment.goal
                )
            next_action = self.reviewer.next_action(
                self.experiment,
                self.lab,
                paths,
                repo_root,
                iteration,
                previous_decision,
                default_target_role=default_task_target,
                default_instructions=default_task_instructions,
            )
            task_target = str(next_action.get("target_role", default_task_target))
            task_instructions = str(next_action.get("instructions", default_task_instructions))
            events.record(
                role="Reviewer Agent",
                event="output",
                message="Reviewer wrote the next action.",
                iteration=iteration,
                data={
                    "artifact": str(paths.reviewer_next_action),
                    "content": next_action,
                },
            )
            if task_target == "STOP":
                iterations.append(
                    {
                        "iteration": iteration,
                        "mode": mode,
                        "decision": Decision.STOP.value,
                        "run_dir": str(paths.run_dir),
                        "reviewer_next_action": str(paths.reviewer_next_action),
                        "reviewer_initial_prompt": str(paths.reviewer_initial_prompt),
                    }
                )
                final_decision = Decision.STOP
                status = "stopped"
                events.record(
                    role="Scheduler",
                    event="stopped",
                    message="Reviewer next action requested STOP before running Executor or Checker.",
                    iteration=iteration,
                )
                break
            if task_target != "Checker Step":
                events.record(
                    role="Reviewer Agent",
                    event="task_assigned",
                    message="Reviewer assigned the implementation task to Executor.",
                    iteration=iteration,
                    data={
                        "target_role": task_target,
                        "instructions": task_instructions,
                        "reviewer_next_action": str(paths.reviewer_next_action),
                        "target_initial_prompt": str(
                            next_action.get(
                                "target_initial_prompt",
                                paths.executor_initial_prompt,
                            )
                        ),
                        "target_readme": str(
                            next_action.get("target_readme", paths.executor_readme)
                        ),
                    },
                )
                events.record(
                    role="Executor Agent",
                    event="wake",
                    message="Executor is running the configured implementation command.",
                    iteration=iteration,
                    data={
                        "command": self.lab.executor_command,
                        "permission": self.lab.executor_permission.value,
                        "initial_prompt": str(paths.executor_initial_prompt),
                        "role_readme": str(paths.executor_readme),
                    },
                )
                self.executor.run(
                    self.experiment,
                    self.lab,
                    paths,
                    repo_root,
                    iteration,
                    previous_decision,
                )
                events.record(
                    role="Executor Agent",
                    event="completed",
                    message="Executor command returned and executor_result.json is available.",
                    iteration=iteration,
                    data={"executor_result": str(paths.executor_result)},
                )
            else:
                events.record(
                    role="Reviewer Agent",
                    event="task_assigned",
                    message="Reviewer assigned the next action to Checker.",
                    iteration=iteration,
                    data={
                        "target_role": task_target,
                        "instructions": task_instructions,
                        "reviewer_next_action": str(paths.reviewer_next_action),
                        "target_initial_prompt": str(
                            next_action.get(
                                "target_initial_prompt",
                                paths.checker_initial_prompt,
                            )
                        ),
                        "target_readme": str(
                            next_action.get("target_readme", paths.checker_readme)
                        ),
                    },
                )
                events.record(
                    role="Executor Agent",
                    event="skipped",
                    message="Reviewer selected Checker, so Executor is skipped for this iteration.",
                    iteration=iteration,
                )
                previous_executor = (
                    run_root / f"iter-{iteration - 1:03d}" / "executor_result.json"
                )
                if previous_executor.exists():
                    paths.executor_result.write_text(
                        previous_executor.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
                else:
                    write_json(
                        paths.executor_result,
                        {
                            "iteration": iteration,
                            "status": "skipped",
                            "summary": "Reviewer selected Checker without running Executor.",
                            "no_diff_reason": "Executor was skipped by Reviewer next action.",
                        },
                    )

            diff = collect_diff(repo_root)
            paths.git_diff.write_text(diff, encoding="utf-8")
            events.record(
                role="Scheduler",
                event="git_diff_collected",
                message="Collected git diff after executor step.",
                iteration=iteration,
                data={"git_diff": str(paths.git_diff), "has_diff": bool(diff.strip())},
            )
            if paths.executor_result.exists():
                ensure_executor_has_diff_explanation(paths.executor_result, diff)

            _hold_checker_until_resources_clean(self.lab, events, iteration)
            events.record(
                role="Checker Step",
                event="wake",
                message="Checker is running the configured benchmark command.",
                iteration=iteration,
                data={
                    "command": self.lab.checker_command,
                    "permission": self.lab.checker_permission.value,
                    "initial_prompt": str(paths.checker_initial_prompt),
                    "role_readme": str(paths.checker_readme),
                },
            )
            self.checker.run(self.lab, paths, repo_root, iteration, previous_decision)
            events.record(
                role="Checker Step",
                event="completed",
                message="Checker command returned and checker_result.json is available.",
                iteration=iteration,
                data={
                    "checker_result": str(paths.checker_result),
                    "summary": _checker_event_summary(paths.checker_result),
                },
            )
            events.record(
                role="Reviewer Agent",
                event="wake",
                message="Reviewer is deciding the next action from executor result, git diff, and checker result.",
                iteration=iteration,
                data={
                    "executor_result": str(paths.executor_result),
                    "git_diff": str(paths.git_diff),
                    "checker_result": str(paths.checker_result),
                    "permission": self.lab.reviewer_permission.value,
                    "initial_prompt": str(paths.reviewer_initial_prompt),
                    "role_readme": str(paths.reviewer_readme),
                },
            )
            decision = self.reviewer.run(
                self.lab,
                paths,
                repo_root,
                iteration,
                previous_decision,
            )
            events.record(
                role="Reviewer Agent",
                event="output",
                message=f"Reviewer wrote decision {decision.decision.value}.",
                iteration=iteration,
                data={
                    "decision": decision.decision.value,
                    "reviewer_decision": str(paths.reviewer_decision),
                    "content": _json_or_decision(paths.reviewer_decision, decision),
                },
            )

            iterations.append(
                {
                    "iteration": iteration,
                    "mode": mode,
                    "decision": decision.decision.value,
                    "run_dir": str(paths.run_dir),
                    "reviewer_next_action": str(paths.reviewer_next_action),
                    "reviewer_initial_prompt": str(paths.reviewer_initial_prompt),
                    "executor_initial_prompt": str(paths.executor_initial_prompt),
                    "checker_initial_prompt": str(paths.checker_initial_prompt),
                    "git_diff": str(paths.git_diff),
                    "executor_result": str(paths.executor_result),
                    "checker_result": str(paths.checker_result),
                    "reviewer_decision": str(paths.reviewer_decision),
                }
            )

            final_decision = decision.decision
            if decision.decision == Decision.ACCEPT:
                status = "accepted"
                if self.lab.commit_on_accept:
                    accepted_diff = collect_diff(repo_root)
                    if accepted_diff.strip():
                        message = self.lab.commit_message_template.format(
                            experiment_name=self.experiment.name,
                            iteration=iteration,
                        )
                        commit_hash = commit_all(repo_root, message)
                        iterations[-1]["commit"] = commit_hash
                        events.record(
                            role="Scheduler",
                            event="committed",
                            message="Accepted experiment changes were committed.",
                            iteration=iteration,
                            data={"commit": commit_hash},
                        )
                    else:
                        events.record(
                            role="Scheduler",
                            event="commit_skipped",
                            message="Reviewer accepted a verification-only iteration with no source diff.",
                            iteration=iteration,
                            data={"reason": "no experiment diff to commit"},
                        )
                break
            if decision.decision == Decision.REJECT:
                status = "rejected"
                if self.lab.rollback_on_reject:
                    rollback_worktree(repo_root)
                    events.record(
                        role="Scheduler",
                        event="rolled_back",
                        message="Rejected experiment changes were rolled back.",
                        iteration=iteration,
                    )
                break
            if decision.decision == Decision.STOP:
                status = "stopped"
                break

            if decision.decision == Decision.NEEDS_RETEST:
                _hold_reviewer_until_resources_clean(
                    self.lab,
                    events,
                    iteration,
                    paths.checker_result,
                )

            previous_decision = decision
            mode = "RETEST" if decision.decision == Decision.NEEDS_RETEST else "EXECUTE"
        else:
            status = "max_iterations_reached"

        result = LoopResult(
            experiment_name=self.experiment.name,
            status=status,
            decision=final_decision,
            run_dir=str(run_root),
            iterations=iterations,
        )
        write_json(run_root / "loop_result.json", result.to_dict())
        events.record(
            role="Scheduler",
            event="run_finished",
            message=f"Loop finished with status {status}.",
            data={"status": status, "decision": final_decision.value},
        )
        return result

    def _make_run_root(self, repo_root: Path) -> Path:
        runs_dir = Path(self.lab.runs_dir)
        if not runs_dir.is_absolute():
            runs_dir = repo_root / runs_dir
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        safe_name = "".join(
            char if char.isalnum() or char in {"-", "_"} else "-"
            for char in self.experiment.name
        ).strip("-")
        return runs_dir / safe_name / timestamp

    def _paths_for_iteration(
        self,
        run_root: Path,
        experiment_json: Path,
        iteration: int,
        previous_decision: ReviewerDecision | None,
        role_paths: dict[str, Path],
    ) -> AgentPaths:
        run_dir = run_root / f"iter-{iteration:03d}"
        previous_path = None
        if previous_decision is not None and iteration > 1:
            previous_path = run_root / f"iter-{iteration - 1:03d}" / "reviewer_decision.json"
        return AgentPaths(
            run_dir=run_dir,
            experiment_json=experiment_json,
            reviewer_next_action=run_dir / "reviewer_next_action.json",
            reviewer_initial_prompt=role_paths["reviewer_initial_prompt"],
            executor_initial_prompt=role_paths["executor_initial_prompt"],
            checker_initial_prompt=role_paths["checker_initial_prompt"],
            reviewer_readme=role_paths["reviewer_readme"],
            executor_readme=role_paths["executor_readme"],
            checker_readme=role_paths["checker_readme"],
            executor_result=run_dir / "executor_result.json",
            checker_result=run_dir / "checker_result.json",
            reviewer_decision=run_dir / "reviewer_decision.json",
            git_diff=run_dir / "git_diff.patch",
            previous_reviewer_decision=previous_path,
        )


def _checker_event_summary(path: Path) -> dict[str, Any]:
    try:
        data = read_json(path)
    except Exception as exc:  # pragma: no cover - defensive logging path
        return {"error": str(exc)}
    qualitative = data.get("qualitative", {})
    if not isinstance(qualitative, dict):
        qualitative = {}
    overall = qualitative.get("overall", {})
    if not isinstance(overall, dict):
        overall = {}
    return {
        "benchmark_valid": data.get("benchmark_valid"),
        "implementation_valid": data.get("implementation_valid"),
        "verdict": data.get("verdict"),
        "recommendation": data.get("recommendation"),
        "rationale": data.get("rationale"),
        "speed": data.get("speed", {}),
        "quality": data.get("quality", {}),
        "qualitative": {
            "status": qualitative.get("status"),
            "quality_label": overall.get("quality_label"),
            "qualitative_pass": overall.get("qualitative_pass"),
            "summary": overall.get("summary"),
        },
    }


def _hold_reviewer_until_resources_clean(
    lab: LabConfig,
    events: EventLog,
    iteration: int,
    checker_result_path: Path,
) -> None:
    reason = _checker_gpu_retest_reason(checker_result_path) or "reviewer_needs_retest"
    result = _hold_until_resources_clean(
        lab,
        events,
        iteration,
        role="Reviewer Agent",
        context="reviewer_retest",
        reason=reason,
    )
    if result.get("status") != "disabled":
        return

    if lab.retest_delay_seconds > 0:
        events.record(
            role="Scheduler",
            event="retest_delay_started",
            message="Delaying before rerunning Checker.",
            iteration=iteration,
            data={"delay_seconds": lab.retest_delay_seconds},
        )
        time.sleep(lab.retest_delay_seconds)
        events.record(
            role="Scheduler",
            event="retest_delay_finished",
            message="Finished delay before rerunning Checker.",
            iteration=iteration,
            data={"delay_seconds": lab.retest_delay_seconds},
        )


def _hold_checker_until_resources_clean(
    lab: LabConfig,
    events: EventLog,
    iteration: int,
) -> None:
    _hold_until_resources_clean(
        lab,
        events,
        iteration,
        role="Checker Step",
        context="checker_preflight",
        reason="before_checker_resource_gate",
    )


def _hold_until_resources_clean(
    lab: LabConfig,
    events: EventLog,
    iteration: int,
    *,
    role: str,
    context: str,
    reason: str,
) -> dict[str, Any]:
    if not lab.resource_hold_enabled:
        return {"status": "disabled"}
    events.record(
        role=role,
        event=f"{context}_resource_hold_started",
        message="Holding until compute resources are clean.",
        iteration=iteration,
        data={
            "reason": reason,
            "timeout_seconds": lab.resource_hold_timeout_seconds,
            "poll_seconds": lab.resource_hold_poll_seconds,
            "threshold_mib": lab.resource_hold_external_memory_threshold_mib,
        },
    )
    result = _wait_for_gpu_idle(
        threshold_mib=lab.resource_hold_external_memory_threshold_mib,
        timeout_seconds=lab.resource_hold_timeout_seconds,
        poll_seconds=lab.resource_hold_poll_seconds,
        events=events,
        role=role,
        iteration=iteration,
        poll_event=f"{context}_resource_hold_poll",
    )
    events.record(
        role=role,
        event=f"{context}_resource_hold_finished",
        message=result["message"],
        iteration=iteration,
        data=result,
    )
    return result


def _resume_state(run_root: Path) -> dict[str, Any]:
    completed_iterations: list[int] = []
    for path in run_root.glob("iter-*"):
        if not path.is_dir():
            continue
        try:
            iteration = int(path.name.removeprefix("iter-"))
        except ValueError:
            continue
        if (path / "reviewer_decision.json").exists():
            completed_iterations.append(iteration)
    if not completed_iterations:
        raise ValueError(f"cannot resume {run_root}: no completed reviewer_decision.json found")
    previous_iteration = max(completed_iterations)
    previous_path = run_root / f"iter-{previous_iteration:03d}" / "reviewer_decision.json"
    previous_decision = ReviewerDecision.from_dict(
        read_json(previous_path),
        iteration=previous_iteration,
    )
    return {
        "previous_iteration": previous_iteration,
        "start_iteration": previous_iteration + 1,
        "previous_decision": previous_decision,
    }


def _checker_gpu_retest_reason(path: Path) -> str:
    try:
        data = read_json(path)
    except Exception:
        return ""
    if str(data.get("recommendation", "")).upper() != Decision.NEEDS_RETEST.value:
        return ""
    artifacts = data.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return ""
    preflight = artifacts.get("gpu_preflight")
    if isinstance(preflight, dict) and preflight.get("status") == "blocked":
        return "gpu_preflight_blocked"
    runtime = artifacts.get("gpu_runtime_monitor")
    if isinstance(runtime, dict) and runtime.get("status") in {"polluted", "blocked"}:
        return "gpu_runtime_polluted"
    rationale = str(data.get("rationale", "")).lower()
    if "gpu preflight" in rationale or "gpu runtime" in rationale:
        return "gpu_rationale"
    return ""


def _wait_for_gpu_idle(
    *,
    threshold_mib: int,
    timeout_seconds: float,
    poll_seconds: float,
    events: EventLog | None = None,
    role: str = "Scheduler",
    iteration: int | None = None,
    poll_event: str = "resource_hold_poll",
) -> dict[str, Any]:
    started = time.monotonic()
    poll = max(poll_seconds, 1.0)
    last_processes: list[dict[str, Any]] = []
    last_error: dict[str, Any] | None = None
    while True:
        probe = _gpu_compute_processes_over_threshold(threshold_mib)
        if probe.get("error_kind") == "missing_tool":
            return {
                "status": "probe_unavailable",
                "message": "nvidia-smi is unavailable; continuing without GPU resource hold.",
                "error": probe["error"],
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
        if probe.get("error"):
            last_error = {
                "error": probe["error"],
                "error_kind": probe.get("error_kind", "probe_failed"),
            }
            last_processes = []
        else:
            last_error = None
            last_processes = probe["processes"]
            if not last_processes:
                return {
                    "status": "idle",
                    "message": "GPU is idle enough for Checker retest.",
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                    "threshold_mib": threshold_mib,
                }
        elapsed = time.monotonic() - started
        if timeout_seconds > 0 and elapsed >= timeout_seconds:
            data: dict[str, Any] = {
                "status": "timeout",
                "message": "GPU resource hold timed out; continuing so Checker can report current state.",
                "elapsed_seconds": round(elapsed, 6),
                "threshold_mib": threshold_mib,
            }
            if last_error is not None:
                data["last_probe_error"] = last_error
            else:
                data["external_compute_processes"] = last_processes
            return data
        sleep_seconds = poll
        if timeout_seconds > 0:
            sleep_seconds = min(poll, max(timeout_seconds - elapsed, 0.0))
        poll_data: dict[str, Any] = {
            "elapsed_seconds": round(elapsed, 6),
            "sleep_seconds": sleep_seconds,
            "threshold_mib": threshold_mib,
        }
        if last_error is not None:
            poll_data["probe_error"] = last_error
            poll_message = "Resource probe failed; sleeping before the next check."
        else:
            poll_data["external_compute_processes"] = last_processes
            poll_message = "Compute resources are busy; sleeping before the next check."
        if events is not None:
            events.record(
                role=role,
                event=poll_event,
                message=poll_message,
                iteration=iteration,
                data=poll_data,
            )
        time.sleep(sleep_seconds)


def _gpu_compute_processes_over_threshold(threshold_mib: int) -> dict[str, Any]:
    visible_bus_ids = _visible_gpu_bus_ids()
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_bus_id,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as exc:
        return {"error": str(exc), "error_kind": "missing_tool", "processes": []}
    except subprocess.TimeoutExpired as exc:
        return {"error": str(exc), "error_kind": "timeout", "processes": []}
    except OSError as exc:
        return {"error": str(exc), "error_kind": "probe_failed", "processes": []}
    if completed.returncode != 0:
        return {
            "error": completed.stderr.strip() or completed.stdout.strip(),
            "error_kind": "command_failed",
            "processes": [],
        }

    processes: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            continue
        gpu_bus_id, pid_text, name, memory_text = parts
        if visible_bus_ids is not None and gpu_bus_id not in visible_bus_ids:
            continue
        try:
            memory_mib = int(memory_text)
        except ValueError:
            continue
        if memory_mib <= threshold_mib:
            continue
        try:
            pid = int(pid_text.strip())
        except ValueError:
            pid = 0
        processes.append(
            {
                "gpu_bus_id": gpu_bus_id,
                "pid": pid,
                "process_name": name,
                "used_gpu_memory_mib": memory_mib,
            }
        )
    return {"processes": processes}


def _visible_gpu_bus_ids() -> set[str] | None:
    import os

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
            check=False,
            capture_output=True,
            text=True,
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


def _json_or_decision(path: Path, decision: ReviewerDecision) -> dict[str, Any]:
    try:
        return read_json(path)
    except Exception:  # pragma: no cover - defensive logging path
        return decision.to_dict()


def run_demo(
    experiment: ExperimentSpec,
    lab: LabConfig | None = None,
    *,
    workdir: str | Path = ".",
) -> Path:
    base = Path(workdir).resolve()
    run_root = _make_demo_run_root(base, experiment.name)
    events = EventLog(run_root / "events.jsonl")
    experiment_json = run_root / "experiment.json"
    write_json(experiment_json, experiment.to_dict())
    role_paths = _write_initial_prompts(run_root, base, lab)

    events.record(
        role="Scheduler",
        event="demo_started",
        message="Started a no-op demo trace. No code, git state, or benchmark is changed.",
        data={"experiment_name": experiment.name, "run_dir": str(run_root)},
    )
    events.record(
        role="Scheduler",
        event="initial_prompts_written",
        message="Wrote role initial prompts that point each agent to its README.",
        data={key: str(value) for key, value in role_paths.items()},
    )
    events.record(
        role="Reviewer Agent",
        event="wake",
        message="Would read ExperimentSpec and choose the next action.",
        iteration=1,
        data={
            "experiment_json": str(experiment_json),
            "initial_prompt": str(role_paths["reviewer_initial_prompt"]),
            "role_readme": str(role_paths["reviewer_readme"]),
            "permission": (
                lab.reviewer_permission.value
                if lab is not None
                else "auto_review"
            ),
        },
    )
    reviewer_next_action = run_root / "iter-001" / "reviewer_next_action.json"
    reviewer_next_action_content = {
        "iteration": 1,
        "target_role": "Executor Agent",
        "instructions": experiment.instructions or experiment.goal,
        "previous_decision": "",
        "reviewer_initial_prompt": str(role_paths["reviewer_initial_prompt"]),
        "reviewer_readme": str(role_paths["reviewer_readme"]),
        "target_initial_prompt": str(role_paths["executor_initial_prompt"]),
        "target_readme": str(role_paths["executor_readme"]),
    }
    write_json(
        reviewer_next_action,
        reviewer_next_action_content,
    )
    events.record(
        role="Reviewer Agent",
        event="output",
        message="Would write the next action.",
        iteration=1,
        data={
            "artifact": str(reviewer_next_action),
            "content": reviewer_next_action_content,
        },
    )
    events.record(
        role="Reviewer Agent",
        event="task_assigned",
        message="Would assign the implementation task to Executor.",
        iteration=1,
        data={
            "target_role": "Executor Agent",
            "instructions": experiment.instructions or experiment.goal,
            "reviewer_next_action": str(reviewer_next_action),
            "target_initial_prompt": str(role_paths["executor_initial_prompt"]),
            "target_readme": str(role_paths["executor_readme"]),
        },
    )
    events.record(
        role="Executor Agent",
        event="wake",
        message="Would implement the task assigned by Reviewer.",
        iteration=1,
        data={
            "experiment_json": str(experiment_json),
            "initial_prompt": str(role_paths["executor_initial_prompt"]),
            "role_readme": str(role_paths["executor_readme"]),
            "permission": (
                lab.executor_permission.value
                if lab is not None
                else "auto_review"
            ),
        },
    )
    events.record(
        role="Executor Agent",
        event="would_write",
        message="Would write executor_result.json and explain no-diff if no code changed.",
        iteration=1,
        data={"artifact": str(run_root / "iter-001" / "executor_result.json")},
    )
    events.record(
        role="Scheduler",
        event="would_collect_git_diff",
        message="Would collect git diff after executor completion.",
        iteration=1,
        data={"artifact": str(run_root / "iter-001" / "git_diff.patch")},
    )
    events.record(
        role="Checker Step",
        event="wake",
        message="Would run benchmark/correctness command without intentionally modifying code.",
        iteration=1,
        data={
            "command": lab.checker_command if lab is not None else "",
            "initial_prompt": str(role_paths["checker_initial_prompt"]),
            "role_readme": str(role_paths["checker_readme"]),
            "permission": (
                lab.checker_permission.value
                if lab is not None
                else "auto_review"
            ),
        },
    )
    events.record(
        role="Checker Step",
        event="would_write",
        message="Would save stdout, stderr, exit code, and checker_result.json.",
        iteration=1,
        data={"artifact": str(run_root / "iter-001" / "checker_result.json")},
    )
    events.record(
        role="Reviewer Agent",
        event="wake",
        message="Would read executor result, git diff, and checker result, then choose the next action.",
        iteration=1,
        data={
            "initial_prompt": str(role_paths["reviewer_initial_prompt"]),
            "role_readme": str(role_paths["reviewer_readme"]),
            "permission": (
                lab.reviewer_permission.value
                if lab is not None
                else "auto_review"
            )
        },
    )
    reviewer_decision = run_root / "iter-001" / "reviewer_decision.json"
    reviewer_decision_content = {
        "iteration": 1,
        "decision": "STOP",
        "rationale": "Demo trace only; no benchmark was run.",
        "next_instructions": "",
    }
    write_json(reviewer_decision, reviewer_decision_content)
    events.record(
        role="Reviewer Agent",
        event="output",
        message="Would write a Reviewer decision.",
        iteration=1,
        data={
            "artifact": str(reviewer_decision),
            "content": reviewer_decision_content,
        },
    )
    events.record(
        role="Scheduler",
        event="demo_finished",
        message="Finished no-op demo trace.",
        data={"events": str(run_root / "events.jsonl")},
    )
    write_json(
        run_root / "demo_result.json",
        {
            "experiment_name": experiment.name,
            "run_dir": str(run_root),
            "events": str(run_root / "events.jsonl"),
        },
    )
    return run_root


def _write_initial_prompts(
    run_root: Path,
    workdir: Path,
    lab: LabConfig | None,
) -> dict[str, Path]:
    prompt_dir = run_root / "initial_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    readmes = {
        "reviewer": _resolve_role_readme(
            workdir,
            lab.reviewer_readme if lab is not None else "README_REVIEWER.md",
        ),
        "executor": _resolve_role_readme(
            workdir,
            lab.executor_readme if lab is not None else "README_EXECUTOR.md",
        ),
        "checker": _resolve_role_readme(
            workdir,
            lab.checker_readme if lab is not None else "README_CHECKER.md",
        ),
    }
    role_labels = {
        "reviewer": "Reviewer Agent",
        "executor": "Executor Agent",
        "checker": "Checker Step",
    }
    results: dict[str, Path] = {}
    for role in ("reviewer", "executor", "checker"):
        prompt_path = prompt_dir / f"{role}_initial_prompt.md"
        readme_path = readmes[role]
        prompt_path.write_text(
            _initial_prompt_text(
                role_label=role_labels[role],
                base_prompt=load_prompt(role),
                readme_path=readme_path,
            ),
            encoding="utf-8",
        )
        results[f"{role}_initial_prompt"] = prompt_path
        results[f"{role}_readme"] = readme_path
    return results


def _resolve_role_readme(workdir: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    if not path.is_absolute():
        path = workdir / path
    return path.resolve()


def _initial_prompt_text(
    *,
    role_label: str,
    base_prompt: str,
    readme_path: Path,
) -> str:
    readme_status = "exists" if readme_path.exists() else "missing"
    return (
        f"# {role_label} Initial Prompt\n\n"
        f"{base_prompt.strip()}\n\n"
        "## Role README\n\n"
        "Before acting, read the role README below and follow it as the "
        "repo-specific operating contract. If it is missing, continue with this "
        "base prompt and record that the README was missing.\n\n"
        f"- Path: `{readme_path}`\n"
        f"- Status: `{readme_status}`\n\n"
        "## Required Behavior\n\n"
        "- Stay within your role boundary.\n"
        "- Use the scheduler artifact paths from environment variables.\n"
        "- Preserve source-code integrity unless your role is explicitly allowed to edit.\n"
    )


def _make_demo_run_root(base: Path, experiment_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-"
        for char in experiment_name
    ).strip("-")
    return base / ".eal" / "demo-runs" / safe_name / timestamp
