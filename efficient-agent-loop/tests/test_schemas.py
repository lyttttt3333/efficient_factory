import json

import pytest

from efficient_agent_loop.schemas import (
    Decision,
    ExperimentSpec,
    LabConfig,
    ReviewerDecision,
    SchemaError,
    read_lab,
)


def test_experiment_spec_requires_name_and_goal():
    with pytest.raises(SchemaError):
        ExperimentSpec.from_dict({"name": "missing-goal"})


def test_experiment_spec_defaults_optional_fields():
    spec = ExperimentSpec.from_dict({"name": "demo", "goal": "change code"})

    assert spec.instructions == ""
    assert spec.target_files == []
    assert spec.constraints == []
    assert spec.success_criteria == []
    assert spec.metadata == {}


def test_lab_config_validates_commands():
    lab = LabConfig.from_dict(
        {
            "executor_command": "python executor.py",
            "checker_command": "python benchmark.py",
            "reviewer_command": "python reviewer.py",
            "max_iterations": 2,
        }
    )

    assert lab.max_iterations == 2
    assert lab.reviewer_command == "python reviewer.py"
    assert lab.rollback_on_reject is True
    assert lab.commit_on_accept is False
    assert lab.executor_permission.value == "auto_review"
    assert lab.checker_permission.value == "auto_review"
    assert lab.reviewer_permission.value == "auto_review"
    assert lab.executor_readme == "README_EXECUTOR.md"
    assert lab.checker_readme == "README_CHECKER.md"
    assert lab.reviewer_readme == "README_REVIEWER.md"
    assert lab.resource_hold_enabled is False
    assert lab.resource_hold_timeout_seconds == 0.0
    assert lab.resource_hold_poll_seconds == 300.0
    assert lab.resource_hold_external_memory_threshold_mib == 1024
    assert lab.retest_delay_seconds == 0.0
    assert lab.gpu_retest_wait_timeout_seconds == 0.0
    assert lab.gpu_retest_wait_poll_seconds == 300.0
    assert lab.gpu_retest_external_memory_threshold_mib == 1024


def test_lab_config_parses_retest_wait_settings():
    lab = LabConfig.from_dict(
        {
            "executor_command": "python executor.py",
            "checker_command": "python benchmark.py",
            "resource_hold_enabled": True,
            "resource_hold_timeout_seconds": 0,
            "resource_hold_poll_seconds": 300,
            "resource_hold_external_memory_threshold_mib": 3072,
            "retest_delay_seconds": 5,
            "gpu_retest_wait_timeout_seconds": 60,
            "gpu_retest_wait_poll_seconds": 2.5,
            "gpu_retest_external_memory_threshold_mib": 2048,
        }
    )

    assert lab.resource_hold_enabled is True
    assert lab.resource_hold_timeout_seconds == 0.0
    assert lab.resource_hold_poll_seconds == 300.0
    assert lab.resource_hold_external_memory_threshold_mib == 3072
    assert lab.retest_delay_seconds == 5.0
    assert lab.gpu_retest_wait_timeout_seconds == 60.0
    assert lab.gpu_retest_wait_poll_seconds == 2.5
    assert lab.gpu_retest_external_memory_threshold_mib == 2048


def test_lab_config_uses_legacy_gpu_retest_settings_as_resource_hold_defaults():
    lab = LabConfig.from_dict(
        {
            "executor_command": "python executor.py",
            "checker_command": "python benchmark.py",
            "resource_hold_enabled": True,
            "gpu_retest_wait_timeout_seconds": 120,
            "gpu_retest_wait_poll_seconds": 5,
            "gpu_retest_external_memory_threshold_mib": 4096,
        }
    )

    assert lab.resource_hold_enabled is True
    assert lab.resource_hold_timeout_seconds == 120.0
    assert lab.resource_hold_poll_seconds == 5.0
    assert lab.resource_hold_external_memory_threshold_mib == 4096


def test_lab_config_allows_unlimited_iteration_sentinel():
    lab = LabConfig.from_dict(
        {
            "executor_command": "python executor.py",
            "checker_command": "python benchmark.py",
            "max_iterations": 0,
        }
    )

    assert lab.max_iterations == 0


def test_reviewer_decision_validates_enum():
    decision = ReviewerDecision.from_dict(
        {"decision": "ACCEPT", "rationale": "passed"},
        iteration=1,
    )

    assert decision.decision == Decision.ACCEPT
    assert decision.iteration == 1


def test_read_lab_supports_simple_yaml_without_pyyaml(tmp_path):
    lab_path = tmp_path / "lab.yaml"
    lab_path.write_text(
        "\n".join(
            [
                "executor_command: 'python executor.py'",
                "checker_command: 'python benchmark.py'",
                "max_iterations: 1",
                "commit_on_accept: false",
            ]
        ),
        encoding="utf-8",
    )

    lab = read_lab(lab_path)

    assert lab.executor_command == "python executor.py"
    assert lab.checker_command == "python benchmark.py"
    assert lab.max_iterations == 1


def test_schema_objects_are_json_serializable():
    spec = ExperimentSpec.from_dict({"name": "demo", "goal": "change code"})

    assert json.loads(json.dumps(spec.to_dict()))["name"] == "demo"
