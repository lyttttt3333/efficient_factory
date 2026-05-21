from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


class SchemaError(ValueError):
    """Raised when user supplied experiment or lab configuration is invalid."""


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    NEEDS_FIX = "NEEDS_FIX"
    NEEDS_RETEST = "NEEDS_RETEST"
    STOP = "STOP"


class CheckerVerdict(str, Enum):
    VALID_EFFECTIVE = "VALID_EFFECTIVE"
    VALID_INEFFECTIVE = "VALID_INEFFECTIVE"
    BENCHMARK_POLLUTED = "BENCHMARK_POLLUTED"
    BENCHMARK_INVALID = "BENCHMARK_INVALID"
    IMPLEMENTATION_INVALID = "IMPLEMENTATION_INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"


class RolePermission(str, Enum):
    AUTO_REVIEW = "auto_review"
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    FULL_ACCESS = "full_access"


@dataclass(slots=True)
class ExperimentSpec:
    name: str
    goal: str
    instructions: str = ""
    target_files: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "ExperimentSpec":
        if not isinstance(data, dict):
            raise SchemaError("ExperimentSpec must be a JSON object.")
        name = _required_str(data, "name")
        goal = _required_str(data, "goal")
        return cls(
            name=name,
            goal=goal,
            instructions=_optional_str(data, "instructions"),
            target_files=_optional_str_list(data, "target_files"),
            constraints=_optional_str_list(data, "constraints"),
            success_criteria=_optional_str_list(data, "success_criteria"),
            metadata=_optional_dict(data, "metadata"),
        )

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class LabConfig:
    executor_command: str
    checker_command: str
    reviewer_command: str = ""
    workdir: str = "."
    runs_dir: str = ".eal/runs"
    max_iterations: int = 3
    commit_on_accept: bool = False
    rollback_on_reject: bool = True
    commit_message_template: str = "efficient-agent-loop: {experiment_name}"
    executor_permission: RolePermission = RolePermission.AUTO_REVIEW
    checker_permission: RolePermission = RolePermission.AUTO_REVIEW
    reviewer_permission: RolePermission = RolePermission.AUTO_REVIEW
    executor_readme: str = "README_EXECUTOR.md"
    checker_readme: str = "README_CHECKER.md"
    reviewer_readme: str = "README_REVIEWER.md"
    resource_hold_enabled: bool = False
    resource_hold_timeout_seconds: float = 0.0
    resource_hold_poll_seconds: float = 300.0
    resource_hold_external_memory_threshold_mib: int = 1024
    retest_delay_seconds: float = 0.0
    gpu_retest_wait_timeout_seconds: float = 0.0
    gpu_retest_wait_poll_seconds: float = 300.0
    gpu_retest_external_memory_threshold_mib: int = 1024

    @classmethod
    def from_dict(cls, data: JsonDict) -> "LabConfig":
        if not isinstance(data, dict):
            raise SchemaError("LabConfig must be a JSON/YAML object.")
        max_iterations = int(data.get("max_iterations", 3))
        if max_iterations < 0:
            raise SchemaError("max_iterations must be >= 0; use 0 for no scheduler iteration limit.")
        legacy_timeout = _optional_nonnegative_float(
            data,
            "gpu_retest_wait_timeout_seconds",
            0.0,
        )
        legacy_poll = _optional_nonnegative_float(
            data,
            "gpu_retest_wait_poll_seconds",
            300.0,
        )
        legacy_threshold = _optional_nonnegative_int(
            data,
            "gpu_retest_external_memory_threshold_mib",
            1024,
        )
        resource_timeout = _optional_nonnegative_float(
            data,
            "resource_hold_timeout_seconds",
            legacy_timeout,
        )
        resource_poll = _optional_nonnegative_float(
            data,
            "resource_hold_poll_seconds",
            legacy_poll,
        )
        resource_threshold = _optional_nonnegative_int(
            data,
            "resource_hold_external_memory_threshold_mib",
            legacy_threshold,
        )
        return cls(
            executor_command=_required_str(data, "executor_command"),
            checker_command=_required_str(data, "checker_command"),
            reviewer_command=_optional_str(data, "reviewer_command"),
            workdir=_optional_str(data, "workdir", "."),
            runs_dir=_optional_str(data, "runs_dir", ".eal/runs"),
            max_iterations=max_iterations,
            commit_on_accept=_optional_bool(data, "commit_on_accept", False),
            rollback_on_reject=_optional_bool(data, "rollback_on_reject", True),
            commit_message_template=_optional_str(
                data,
                "commit_message_template",
                "efficient-agent-loop: {experiment_name}",
            ),
            executor_permission=_optional_permission(
                data,
                "executor_permission",
                RolePermission.AUTO_REVIEW,
            ),
            checker_permission=_optional_permission(
                data,
                "checker_permission",
                RolePermission.AUTO_REVIEW,
            ),
            reviewer_permission=_optional_permission(
                data,
                "reviewer_permission",
                RolePermission.AUTO_REVIEW,
            ),
            executor_readme=_optional_str(data, "executor_readme", "README_EXECUTOR.md"),
            checker_readme=_optional_str(data, "checker_readme", "README_CHECKER.md"),
            reviewer_readme=_optional_str(data, "reviewer_readme", "README_REVIEWER.md"),
            resource_hold_enabled=_optional_bool(
                data,
                "resource_hold_enabled",
                False,
            ),
            resource_hold_timeout_seconds=resource_timeout,
            resource_hold_poll_seconds=resource_poll,
            resource_hold_external_memory_threshold_mib=resource_threshold,
            retest_delay_seconds=_optional_nonnegative_float(
                data,
                "retest_delay_seconds",
                0.0,
            ),
            gpu_retest_wait_timeout_seconds=legacy_timeout,
            gpu_retest_wait_poll_seconds=legacy_poll,
            gpu_retest_external_memory_threshold_mib=legacy_threshold,
        )

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        data["executor_permission"] = self.executor_permission.value
        data["checker_permission"] = self.checker_permission.value
        data["reviewer_permission"] = self.reviewer_permission.value
        return data


@dataclass(slots=True)
class CommandResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class ExecutorResult:
    iteration: int
    status: str
    summary: str
    command: CommandResult
    no_diff_reason: str | None = None
    artifacts: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class CheckerResult:
    iteration: int
    status: str
    passed: bool
    command: CommandResult
    git_diff_unchanged: bool
    benchmark_valid: bool
    implementation_valid: bool
    verdict: CheckerVerdict
    recommendation: Decision
    rationale: str = ""
    speed: JsonDict = field(default_factory=dict)
    quality: JsonDict = field(default_factory=dict)
    qualitative: JsonDict = field(default_factory=dict)
    pollution_detected: bool = False
    artifacts: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        data["recommendation"] = self.recommendation.value
        return data


@dataclass(slots=True)
class ReviewerDecision:
    iteration: int
    decision: Decision
    rationale: str
    next_instructions: str = ""

    @classmethod
    def from_dict(cls, data: JsonDict, iteration: int | None = None) -> "ReviewerDecision":
        if not isinstance(data, dict):
            raise SchemaError("Reviewer decision must be a JSON object.")
        raw_decision = _required_str(data, "decision")
        try:
            decision = Decision(raw_decision)
        except ValueError as exc:
            valid = ", ".join(item.value for item in Decision)
            raise SchemaError(f"decision must be one of: {valid}") from exc
        return cls(
            iteration=int(data.get("iteration", iteration or 0)),
            decision=decision,
            rationale=_required_str(data, "rationale"),
            next_instructions=_optional_str(data, "next_instructions"),
        )

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data


@dataclass(slots=True)
class LoopResult:
    experiment_name: str
    status: str
    decision: Decision
    run_dir: str
    iterations: list[JsonDict] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        data["decision"] = self.decision.value
        return data


def read_experiment(path: str | Path) -> ExperimentSpec:
    return ExperimentSpec.from_dict(read_json(path))


def read_lab(path: str | Path) -> LabConfig:
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = read_yaml(path)
    else:
        data = read_json(path)
    return LabConfig.from_dict(data)


def read_json(path: str | Path) -> JsonDict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SchemaError(f"{path} must contain a JSON object.")
    return data


def write_json(path: str | Path, data: JsonDict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_yaml(path: str | Path) -> JsonDict:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        return _read_simple_yaml(text)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SchemaError(f"{path} must contain a YAML object.")
    return data


def _read_simple_yaml(text: str) -> JsonDict:
    data: JsonDict = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if raw_line.startswith((" ", "\t")):
            if current_key is None:
                raise SchemaError("Invalid YAML continuation without a key.")
            stripped = raw_line.strip()
            if not stripped.startswith("- "):
                raise SchemaError("Fallback YAML parser only supports simple lists.")
            data.setdefault(current_key, []).append(_parse_scalar(stripped[2:].strip()))
            continue
        key, sep, value = raw_line.partition(":")
        if not sep:
            raise SchemaError("Fallback YAML parser only supports key: value entries.")
        current_key = key.strip()
        value = value.strip()
        data[current_key] = [] if value == "" else _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        pass
    try:
        return int(value)
    except ValueError:
        return value


def _required_str(data: JsonDict, key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{key} is required and must be a non-empty string.")
    return value


def _optional_str(data: JsonDict, key: str, default: str = "") -> str:
    value = data.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise SchemaError(f"{key} must be a string.")
    return value


def _optional_str_list(data: JsonDict, key: str) -> list[str]:
    value = data.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SchemaError(f"{key} must be a list of strings.")
    return value


def _optional_dict(data: JsonDict, key: str) -> JsonDict:
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SchemaError(f"{key} must be an object.")
    return value


def _optional_bool(data: JsonDict, key: str, default: bool) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise SchemaError(f"{key} must be a boolean.")


def _optional_nonnegative_float(data: JsonDict, key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool):
        raise SchemaError(f"{key} must be a non-negative number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{key} must be a non-negative number.") from exc
    if number < 0:
        raise SchemaError(f"{key} must be a non-negative number.")
    return number


def _optional_nonnegative_int(data: JsonDict, key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool):
        raise SchemaError(f"{key} must be a non-negative integer.")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{key} must be a non-negative integer.") from exc
    if number < 0:
        raise SchemaError(f"{key} must be a non-negative integer.")
    return number


def _optional_permission(
    data: JsonDict,
    key: str,
    default: RolePermission,
) -> RolePermission:
    value = data.get(key, default.value)
    if isinstance(value, RolePermission):
        return value
    if not isinstance(value, str):
        raise SchemaError(f"{key} must be a permission string.")
    try:
        return RolePermission(value)
    except ValueError as exc:
        valid = ", ".join(item.value for item in RolePermission)
        raise SchemaError(f"{key} must be one of: {valid}") from exc
