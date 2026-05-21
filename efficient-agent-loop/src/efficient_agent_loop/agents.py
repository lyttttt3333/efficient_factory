from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from . import git_utils
from .schemas import (
    CheckerResult,
    CheckerVerdict,
    CommandResult,
    Decision,
    ExecutorResult,
    ExperimentSpec,
    JsonDict,
    LabConfig,
    ReviewerDecision,
    SchemaError,
    read_json,
    write_json,
)


@dataclass(slots=True)
class AgentPaths:
    run_dir: Path
    experiment_json: Path
    reviewer_next_action: Path
    reviewer_initial_prompt: Path
    executor_initial_prompt: Path
    checker_initial_prompt: Path
    reviewer_readme: Path
    executor_readme: Path
    checker_readme: Path
    executor_result: Path
    checker_result: Path
    reviewer_decision: Path
    git_diff: Path
    previous_reviewer_decision: Path | None = None


def run_shell_command(command: str, cwd: Path, env: dict[str, str]) -> CommandResult:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        text=True,
        capture_output=True,
        env={**os.environ, **env},
        check=False,
    )
    return CommandResult(
        command=command,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_seconds=round(time.monotonic() - started, 6),
    )


class ExecutorAgent:
    def run(
        self,
        experiment: ExperimentSpec,
        lab: LabConfig,
        paths: AgentPaths,
        workdir: Path,
        iteration: int,
        previous_decision: ReviewerDecision | None,
    ) -> ExecutorResult:
        env = build_agent_env(
            paths,
            workdir,
            iteration,
            previous_decision,
            role_permission=lab.executor_permission.value,
            agent_role="executor",
        )
        command_result = run_shell_command(lab.executor_command, workdir, env)

        if paths.executor_result.exists():
            data = read_json(paths.executor_result)
            data.setdefault("iteration", iteration)
            data.setdefault("status", "completed" if command_result.exit_code == 0 else "failed")
            data.setdefault("summary", "Executor command completed.")
            data.setdefault("command", command_result.to_dict())
            data.setdefault("no_diff_reason", None)
            write_json(paths.executor_result, data)
            return _executor_from_json(data, command_result, iteration)

        result = ExecutorResult(
            iteration=iteration,
            status="completed" if command_result.exit_code == 0 else "failed",
            summary=(
                "Executor command completed without writing executor_result.json."
                if command_result.exit_code == 0
                else "Executor command failed before writing executor_result.json."
            ),
            command=command_result,
            artifacts={"experiment_name": experiment.name},
        )
        write_json(paths.executor_result, result.to_dict())
        return result


class CheckerStep:
    def run(
        self,
        lab: LabConfig,
        paths: AgentPaths,
        workdir: Path,
        iteration: int,
        previous_decision: ReviewerDecision | None,
    ) -> CheckerResult:
        before_diff = git_utils.collect_diff(workdir)
        env = build_agent_env(
            paths,
            workdir,
            iteration,
            previous_decision,
            role_permission=lab.checker_permission.value,
            agent_role="checker",
        )
        command_result = run_shell_command(lab.checker_command, workdir, env)
        after_diff = git_utils.collect_diff(workdir)
        pollution_detected = before_diff != after_diff
        data = read_json(paths.checker_result) if paths.checker_result.exists() else {}

        benchmark_valid = bool(data.get("benchmark_valid", not pollution_detected))
        implementation_valid = bool(
            data.get(
                "implementation_valid",
                command_result.exit_code == 0 and benchmark_valid,
            )
        )
        if pollution_detected:
            benchmark_valid = False
            implementation_valid = False
            verdict = CheckerVerdict.BENCHMARK_POLLUTED
            recommendation = Decision.NEEDS_RETEST
            rationale = (
                "Checker changed the git diff while running benchmark, so the "
                "benchmark result is polluted."
            )
        else:
            verdict = _checker_verdict_from_json(
                data,
                command_result.exit_code,
                benchmark_valid,
                implementation_valid,
            )
            recommendation = _checker_recommendation_from_json(
                data,
                benchmark_valid,
                implementation_valid,
            )
            rationale = str(
                data.get(
                    "rationale",
                    "Benchmark completed and implementation validity was inferred from exit code.",
                )
            )

        result = CheckerResult(
            iteration=int(data.get("iteration", iteration)),
            status=str(data.get("status", "completed")),
            passed=benchmark_valid and implementation_valid,
            command=command_result,
            git_diff_unchanged=not pollution_detected,
            benchmark_valid=benchmark_valid,
            implementation_valid=implementation_valid,
            verdict=verdict,
            recommendation=recommendation,
            rationale=rationale,
            speed=_json_dict(data.get("speed", {}), "speed"),
            quality=_json_dict(data.get("quality", {}), "quality"),
            qualitative=_json_dict(data.get("qualitative", {}), "qualitative"),
            pollution_detected=pollution_detected,
            artifacts={
                **_json_dict(data.get("artifacts", {}), "artifacts"),
                "stdout_path": str(paths.checker_result.with_suffix(".stdout.txt")),
                "stderr_path": str(paths.checker_result.with_suffix(".stderr.txt")),
            },
        )
        paths.checker_result.with_suffix(".stdout.txt").write_text(
            command_result.stdout,
            encoding="utf-8",
        )
        paths.checker_result.with_suffix(".stderr.txt").write_text(
            command_result.stderr,
            encoding="utf-8",
        )
        write_json(paths.checker_result, result.to_dict())
        return result


class ReviewerAgent:
    def next_action(
        self,
        experiment: ExperimentSpec,
        lab: LabConfig,
        paths: AgentPaths,
        workdir: Path,
        iteration: int,
        previous_decision: ReviewerDecision | None,
        *,
        default_target_role: str,
        default_instructions: str,
    ) -> JsonDict:
        if lab.reviewer_command.strip():
            return self._run_external_next_action(
                lab,
                paths,
                workdir,
                iteration,
                previous_decision,
                default_target_role=default_target_role,
                default_instructions=default_instructions,
            )
        action = _default_reviewer_next_action(
            iteration=iteration,
            target_role=default_target_role,
            instructions=default_instructions,
            previous_decision=previous_decision,
            paths=paths,
            source="builtin",
        )
        write_json(paths.reviewer_next_action, action)
        return action

    def _run_external_next_action(
        self,
        lab: LabConfig,
        paths: AgentPaths,
        workdir: Path,
        iteration: int,
        previous_decision: ReviewerDecision | None,
        *,
        default_target_role: str,
        default_instructions: str,
    ) -> JsonDict:
        before_diff = git_utils.collect_diff(workdir)
        env = build_agent_env(
            paths,
            workdir,
            iteration,
            previous_decision,
            role_permission=lab.reviewer_permission.value,
            agent_role="reviewer",
            reviewer_context="before_executor",
        )
        command_result = run_shell_command(lab.reviewer_command, workdir, env)
        after_diff = git_utils.collect_diff(workdir)
        paths.reviewer_next_action.with_suffix(".stdout.txt").write_text(
            command_result.stdout,
            encoding="utf-8",
        )
        paths.reviewer_next_action.with_suffix(".stderr.txt").write_text(
            command_result.stderr,
            encoding="utf-8",
        )

        if paths.reviewer_next_action.exists():
            action = read_json(paths.reviewer_next_action)
        else:
            action = _default_reviewer_next_action(
                iteration=iteration,
                target_role=default_target_role,
                instructions=default_instructions,
                previous_decision=previous_decision,
                paths=paths,
                source="fallback_after_reviewer_command_missing_next_action",
            )
            action["reviewer_error"] = (
                "reviewer_command did not write reviewer_next_action.json"
            )

        action.setdefault("iteration", iteration)
        action.setdefault("target_role", default_target_role)
        action.setdefault("instructions", default_instructions)
        action.setdefault(
            "previous_decision",
            previous_decision.decision.value if previous_decision is not None else "",
        )
        action.setdefault("source", "reviewer_command")
        action.setdefault("reviewer_initial_prompt", str(paths.reviewer_initial_prompt))
        action.setdefault("reviewer_readme", str(paths.reviewer_readme))
        action.setdefault(
            "target_initial_prompt",
            str(
                paths.checker_initial_prompt
                if action.get("target_role") == "Checker Step"
                else paths.executor_initial_prompt
            ),
        )
        action.setdefault(
            "target_readme",
            str(
                paths.checker_readme
                if action.get("target_role") == "Checker Step"
                else paths.executor_readme
            ),
        )
        action["command"] = command_result.to_dict()
        action["git_diff_unchanged"] = before_diff == after_diff
        if before_diff != after_diff:
            action["reviewer_error"] = (
                "reviewer_command modified the git diff, which violates reviewer constraints."
            )
        _validate_reviewer_next_action(action)
        write_json(paths.reviewer_next_action, action)
        return action

    def run(
        self,
        lab: LabConfig,
        paths: AgentPaths,
        workdir: Path,
        iteration: int,
        previous_decision: ReviewerDecision | None,
    ) -> ReviewerDecision:
        if lab.reviewer_command.strip():
            return self._run_external(lab, paths, workdir, iteration, previous_decision)
        decision = self._run_builtin(paths, iteration)
        write_json(paths.reviewer_decision, decision.to_dict())
        return decision

    def _run_external(
        self,
        lab: LabConfig,
        paths: AgentPaths,
        workdir: Path,
        iteration: int,
        previous_decision: ReviewerDecision | None,
    ) -> ReviewerDecision:
        before_diff = git_utils.collect_diff(workdir)
        env = build_agent_env(
            paths,
            workdir,
            iteration,
            previous_decision,
            role_permission=lab.reviewer_permission.value,
            agent_role="reviewer",
            reviewer_context="after_checker",
        )
        command_result = run_shell_command(lab.reviewer_command, workdir, env)
        after_diff = git_utils.collect_diff(workdir)

        if not paths.reviewer_decision.exists():
            decision = ReviewerDecision(
                iteration=iteration,
                decision=Decision.STOP,
                rationale=(
                    "Reviewer command did not write reviewer_decision.json. "
                    f"exit_code={command_result.exit_code}."
                ),
            )
            write_json(paths.reviewer_decision, decision.to_dict())
            return decision

        data = read_json(paths.reviewer_decision)
        if before_diff != after_diff:
            data["decision"] = Decision.STOP.value
            data["rationale"] = (
                data.get("rationale", "")
                + "\nReviewer command modified the git diff, which violates reviewer constraints."
            ).strip()
            write_json(paths.reviewer_decision, data)
        return ReviewerDecision.from_dict(data, iteration=iteration)

    def _run_builtin(self, paths: AgentPaths, iteration: int) -> ReviewerDecision:
        executor = read_json(paths.executor_result)
        checker = read_json(paths.checker_result)
        diff = paths.git_diff.read_text(encoding="utf-8")
        command = executor.get("command", {})
        executor_exit = command.get("exit_code", 0)
        checker_command = checker.get("command", {})
        checker_exit = checker_command.get("exit_code", 1)
        diff_present = bool(diff.strip())
        no_diff_reason = str(executor.get("no_diff_reason", "")).strip()
        no_diff_reason_source = str(executor.get("no_diff_reason_source", "")).strip()

        if executor_exit != 0:
            return ReviewerDecision(
                iteration=iteration,
                decision=Decision.NEEDS_FIX,
                rationale=f"Executor command failed with exit code {executor_exit}.",
                next_instructions="Fix the executor implementation failure and try again.",
            )
        if not diff_present and (
            not no_diff_reason or no_diff_reason_source == "scheduler_fallback"
        ):
            return ReviewerDecision(
                iteration=iteration,
                decision=Decision.NEEDS_FIX,
                rationale=(
                    "Executor produced no git diff and did not provide a specific "
                    "executor-authored explanation."
                ),
                next_instructions=(
                    "Either implement a code change or write a concrete no_diff_reason "
                    "explaining why this iteration is verification-only."
                ),
            )
        if not checker.get("git_diff_unchanged", True):
            return ReviewerDecision(
                iteration=iteration,
                decision=Decision.NEEDS_RETEST,
                rationale="Checker changed the git diff, so the benchmark result is not trusted.",
                next_instructions="Restore checker side effects and rerun the checker.",
            )
        checker_recommendation = checker.get("recommendation")
        if checker_recommendation == Decision.NEEDS_RETEST.value or not checker.get(
            "benchmark_valid",
            checker_exit == 0,
        ):
            return ReviewerDecision(
                iteration=iteration,
                decision=Decision.NEEDS_RETEST,
                rationale=checker.get("rationale", "Checker reported an invalid benchmark."),
                next_instructions="Rerun with an uncontaminated benchmark result.",
            )
        if checker_recommendation == Decision.REJECT.value:
            return ReviewerDecision(
                iteration=iteration,
                decision=Decision.REJECT,
                rationale=checker.get("rationale", "Checker rejected the experiment."),
            )
        if checker_recommendation == Decision.NEEDS_FIX.value:
            return ReviewerDecision(
                iteration=iteration,
                decision=Decision.NEEDS_FIX,
                rationale=checker.get("rationale", "Checker requested implementation fixes."),
                next_instructions="Use checker speed/quality/stdout/stderr to fix the implementation.",
            )
        if checker_recommendation == Decision.STOP.value:
            return ReviewerDecision(
                iteration=iteration,
                decision=Decision.STOP,
                rationale=checker.get("rationale", "Checker requested STOP."),
            )
        if checker.get("implementation_valid", checker_exit == 0):
            rationale = checker.get(
                "rationale",
                "Checker reported a valid benchmark and effective implementation.",
            )
            if not diff_present:
                rationale = (
                    f"{rationale} Executor produced no source diff, but explained "
                    f"the verification-only iteration: {no_diff_reason}"
                )
            return ReviewerDecision(
                iteration=iteration,
                decision=Decision.ACCEPT,
                rationale=rationale,
            )
        return ReviewerDecision(
            iteration=iteration,
            decision=Decision.NEEDS_FIX,
            rationale=checker.get(
                "rationale",
                f"Checker command failed with exit code {checker_exit}.",
            ),
            next_instructions="Use checker speed/quality/stdout/stderr to fix the implementation.",
        )


def build_agent_env(
    paths: AgentPaths,
    workdir: Path,
    iteration: int,
    previous_decision: ReviewerDecision | None,
    role_permission: str = "",
    agent_role: str = "",
    reviewer_context: str = "",
) -> dict[str, str]:
    return {
        "EAL_EXPERIMENT_JSON": str(paths.experiment_json),
        "EAL_WORKDIR": str(workdir),
        "EAL_RUN_DIR": str(paths.run_dir),
        "EAL_ITERATION": str(iteration),
        "EAL_EXECUTOR_RESULT": str(paths.executor_result),
        "EAL_CHECKER_RESULT": str(paths.checker_result),
        "EAL_REVIEWER_DECISION": str(paths.reviewer_decision),
        "EAL_REVIEWER_NEXT_ACTION": str(paths.reviewer_next_action),
        "EAL_REVIEWER_CONTEXT": reviewer_context,
        "EAL_REVIEWER_INITIAL_PROMPT": str(paths.reviewer_initial_prompt),
        "EAL_EXECUTOR_INITIAL_PROMPT": str(paths.executor_initial_prompt),
        "EAL_CHECKER_INITIAL_PROMPT": str(paths.checker_initial_prompt),
        "EAL_REVIEWER_README": str(paths.reviewer_readme),
        "EAL_EXECUTOR_README": str(paths.executor_readme),
        "EAL_CHECKER_README": str(paths.checker_readme),
        "EAL_GIT_DIFF": str(paths.git_diff),
        "EAL_PREVIOUS_REVIEWER_DECISION": (
            str(paths.previous_reviewer_decision)
            if paths.previous_reviewer_decision is not None
            else ""
        ),
        "EAL_PREVIOUS_DECISION": (
            previous_decision.decision.value if previous_decision is not None else ""
        ),
        "EAL_PREVIOUS_INSTRUCTIONS": (
            previous_decision.next_instructions if previous_decision is not None else ""
        ),
        "EAL_AGENT_PERMISSION": role_permission,
        **_agent_prompt_env(paths, agent_role),
    }


def ensure_executor_has_diff_explanation(path: Path, diff: str) -> None:
    data = read_json(path)
    if diff.strip():
        data["git_diff_present"] = True
        write_json(path, data)
        return
    data["git_diff_present"] = False
    if not data.get("no_diff_reason"):
        data["no_diff_reason"] = (
            "Executor command completed but produced no git diff, and the executor "
            "did not provide a more specific explanation."
        )
        data["no_diff_reason_source"] = "scheduler_fallback"
    else:
        data.setdefault("no_diff_reason_source", "executor")
    write_json(path, data)


def _agent_prompt_env(paths: AgentPaths, agent_role: str) -> dict[str, str]:
    if agent_role == "executor":
        return {
            "EAL_AGENT_INITIAL_PROMPT": str(paths.executor_initial_prompt),
            "EAL_AGENT_README": str(paths.executor_readme),
        }
    if agent_role == "checker":
        return {
            "EAL_AGENT_INITIAL_PROMPT": str(paths.checker_initial_prompt),
            "EAL_AGENT_README": str(paths.checker_readme),
        }
    if agent_role == "reviewer":
        return {
            "EAL_AGENT_INITIAL_PROMPT": str(paths.reviewer_initial_prompt),
            "EAL_AGENT_README": str(paths.reviewer_readme),
        }
    return {
        "EAL_AGENT_INITIAL_PROMPT": "",
        "EAL_AGENT_README": "",
    }


def _default_reviewer_next_action(
    *,
    iteration: int,
    target_role: str,
    instructions: str,
    previous_decision: ReviewerDecision | None,
    paths: AgentPaths,
    source: str,
) -> JsonDict:
    return {
        "iteration": iteration,
        "target_role": target_role,
        "instructions": instructions,
        "previous_decision": (
            previous_decision.decision.value if previous_decision is not None else ""
        ),
        "source": source,
        "reviewer_initial_prompt": str(paths.reviewer_initial_prompt),
        "reviewer_readme": str(paths.reviewer_readme),
        "target_initial_prompt": str(
            paths.checker_initial_prompt
            if target_role == "Checker Step"
            else paths.executor_initial_prompt
        ),
        "target_readme": str(
            paths.checker_readme if target_role == "Checker Step" else paths.executor_readme
        ),
    }


def _validate_reviewer_next_action(action: JsonDict) -> None:
    target_role = action.get("target_role")
    if target_role not in {"Executor Agent", "Checker Step", "STOP"}:
        raise SchemaError(
            "reviewer_next_action.json target_role must be Executor Agent, Checker Step, or STOP."
        )
    instructions = action.get("instructions")
    if target_role != "STOP" and (
        not isinstance(instructions, str) or not instructions.strip()
    ):
        raise SchemaError(
            "reviewer_next_action.json instructions must be a non-empty string."
        )


def _executor_from_json(
    data: JsonDict,
    command_result: CommandResult,
    iteration: int,
) -> ExecutorResult:
    command_data = data.get("command")
    if isinstance(command_data, dict):
        command = CommandResult(
            command=str(command_data.get("command", command_result.command)),
            exit_code=int(command_data.get("exit_code", command_result.exit_code)),
            stdout=str(command_data.get("stdout", command_result.stdout)),
            stderr=str(command_data.get("stderr", command_result.stderr)),
            duration_seconds=float(
                command_data.get("duration_seconds", command_result.duration_seconds)
            ),
        )
    else:
        command = command_result
    status = data.get("status", "completed")
    summary = data.get("summary", "Executor command completed.")
    if not isinstance(status, str) or not isinstance(summary, str):
        raise SchemaError("executor_result.json status and summary must be strings.")
    return ExecutorResult(
        iteration=int(data.get("iteration", iteration)),
        status=status,
        summary=summary,
        command=command,
        no_diff_reason=data.get("no_diff_reason"),
        artifacts=data.get("artifacts", {}),
    )


def _checker_verdict_from_json(
    data: JsonDict,
    exit_code: int,
    benchmark_valid: bool,
    implementation_valid: bool,
) -> CheckerVerdict:
    raw = data.get("verdict")
    if isinstance(raw, str):
        try:
            return CheckerVerdict(raw)
        except ValueError as exc:
            valid = ", ".join(item.value for item in CheckerVerdict)
            raise SchemaError(f"checker verdict must be one of: {valid}") from exc
    if not benchmark_valid:
        return CheckerVerdict.BENCHMARK_INVALID
    if implementation_valid and exit_code == 0:
        return CheckerVerdict.VALID_EFFECTIVE
    return CheckerVerdict.IMPLEMENTATION_INVALID


def _checker_recommendation_from_json(
    data: JsonDict,
    benchmark_valid: bool,
    implementation_valid: bool,
) -> Decision:
    raw = data.get("recommendation")
    if isinstance(raw, str):
        try:
            return Decision(raw)
        except ValueError as exc:
            valid = ", ".join(item.value for item in Decision)
            raise SchemaError(f"checker recommendation must be one of: {valid}") from exc
    if not benchmark_valid:
        return Decision.NEEDS_RETEST
    if implementation_valid:
        return Decision.ACCEPT
    return Decision.NEEDS_FIX


def _json_dict(value: object, name: str) -> JsonDict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SchemaError(f"checker_result.json field {name} must be an object.")
    return value
