import json
import subprocess
import sys
from pathlib import Path

import efficient_agent_loop.scheduler as scheduler_module
from efficient_agent_loop.scheduler import (
    EventLog,
    Scheduler,
    _checker_gpu_retest_reason,
    _hold_until_resources_clean,
    run_demo,
)
from efficient_agent_loop.schemas import Decision, ExperimentSpec, LabConfig


def test_scheduler_accepts_successful_executor_and_checker(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    executor_script = scripts / "executor.py"
    executor_script.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "Path('feature.txt').write_text('hello\\n', encoding='utf-8')",
                "Path(os.environ['EAL_EXECUTOR_RESULT']).write_text(",
                "    json.dumps({'iteration': int(os.environ['EAL_ITERATION']), 'status': 'completed', 'summary': 'created feature.txt', 'no_diff_reason': None}, indent=2) + '\\n',",
                "    encoding='utf-8',",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    benchmark = repo / "benchmark.py"
    benchmark.write_text(
        "from pathlib import Path\n"
        "assert Path('feature.txt').read_text(encoding='utf-8').strip() == 'hello'\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    (repo / "README_REVIEWER.md").write_text("reviewer contract\n", encoding="utf-8")
    (repo / "README_EXECUTOR.md").write_text("executor contract\n", encoding="utf-8")
    (repo / "README_CHECKER.md").write_text("checker contract\n", encoding="utf-8")
    _git(
        repo,
        "add",
        "benchmark.py",
        "README_REVIEWER.md",
        "README_EXECUTOR.md",
        "README_CHECKER.md",
    )
    _git(repo, "commit", "-m", "add benchmark")

    scheduler = Scheduler(
        ExperimentSpec.from_dict({"name": "demo", "goal": "create feature"}),
        LabConfig.from_dict(
            {
                "executor_command": f"{sys.executable} {executor_script}",
                "checker_command": f"{sys.executable} benchmark.py",
                "max_iterations": 0,
                "rollback_on_reject": False,
            }
        ),
        workdir=repo,
    )

    result = scheduler.run()

    assert result.decision == Decision.ACCEPT
    assert (repo / "feature.txt").read_text(encoding="utf-8") == "hello\n"
    iteration = result.iterations[0]
    assert Path(iteration["reviewer_next_action"]).exists()
    assert Path(iteration["executor_initial_prompt"]).exists()
    assert Path(iteration["checker_initial_prompt"]).exists()
    assert Path(iteration["git_diff"]).read_text(encoding="utf-8").find("feature.txt") >= 0
    reviewer_next_action = json.loads(
        Path(iteration["reviewer_next_action"]).read_text(encoding="utf-8")
    )
    assert reviewer_next_action["target_role"] == "Executor Agent"
    executor_prompt = Path(iteration["executor_initial_prompt"]).read_text(encoding="utf-8")
    assert "README_EXECUTOR.md" in executor_prompt
    assert "Status: `exists`" in executor_prompt
    checker = json.loads(Path(iteration["checker_result"]).read_text(encoding="utf-8"))
    assert checker["command"]["exit_code"] == 0
    assert checker["git_diff_unchanged"] is True
    assert checker["benchmark_valid"] is True
    assert checker["implementation_valid"] is True
    assert checker["recommendation"] == "ACCEPT"
    events = _read_jsonl(Path(result.run_dir) / "events.jsonl")
    assert [event["role"] for event in events if event["event"] == "wake"] == [
        "Reviewer Agent",
        "Executor Agent",
        "Checker Step",
        "Reviewer Agent",
    ]
    wake_data = [event["data"] for event in events if event["event"] == "wake"]
    assert wake_data[0]["permission"] == "auto_review"
    assert Path(wake_data[0]["initial_prompt"]).exists()
    assert wake_data[1]["permission"] == "auto_review"
    assert Path(wake_data[1]["initial_prompt"]).exists()
    assert wake_data[2]["permission"] == "auto_review"
    assert Path(wake_data[2]["initial_prompt"]).exists()
    assert wake_data[3]["permission"] == "auto_review"
    reviewer_outputs = [
        event["data"]["content"]
        for event in events
        if event["role"] == "Reviewer Agent" and event["event"] == "output"
    ]
    assert reviewer_outputs[0]["target_role"] == "Executor Agent"
    assert reviewer_outputs[1]["decision"] == "ACCEPT"


def test_scheduler_requests_fix_for_unexplained_no_diff(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    executor_script = scripts / "executor_noop.py"
    executor_script.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "Path(os.environ['EAL_EXECUTOR_RESULT']).write_text(",
                "    json.dumps({'iteration': int(os.environ['EAL_ITERATION']), 'status': 'completed', 'summary': 'noop'}, indent=2) + '\\n',",
                "    encoding='utf-8',",
                ")",
            ]
        ),
        encoding="utf-8",
    )

    scheduler = Scheduler(
        ExperimentSpec.from_dict({"name": "noop", "goal": "do nothing"}),
        LabConfig.from_dict(
            {
                "executor_command": f"{sys.executable} {executor_script}",
                "checker_command": f"{sys.executable} -c \"print('checker ran')\"",
                "max_iterations": 1,
                "rollback_on_reject": True,
            }
        ),
        workdir=repo,
    )

    result = scheduler.run()

    assert result.decision == Decision.NEEDS_FIX
    iteration = result.iterations[0]
    executor = json.loads(Path(iteration["executor_result"]).read_text(encoding="utf-8"))
    assert executor["git_diff_present"] is False
    assert "no git diff" in executor["no_diff_reason"]
    assert executor["no_diff_reason_source"] == "scheduler_fallback"
    checker = json.loads(Path(iteration["checker_result"]).read_text(encoding="utf-8"))
    assert checker["command"]["exit_code"] == 0


def test_scheduler_allows_explained_no_diff_to_use_checker_result(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    executor_script = scripts / "executor_noop_explained.py"
    executor_script.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "Path(os.environ['EAL_EXECUTOR_RESULT']).write_text(",
                "    json.dumps({'iteration': int(os.environ['EAL_ITERATION']), 'status': 'completed', 'summary': 'verification only', 'no_diff_reason': 'target behavior already exists and only needs checker verification'}, indent=2) + '\\n',",
                "    encoding='utf-8',",
                ")",
            ]
        ),
        encoding="utf-8",
    )

    scheduler = Scheduler(
        ExperimentSpec.from_dict({"name": "noop-verified", "goal": "verify existing behavior"}),
        LabConfig.from_dict(
            {
                "executor_command": f"{sys.executable} {executor_script}",
                "checker_command": f"{sys.executable} -c \"print('checker passed')\"",
                "max_iterations": 1,
                "rollback_on_reject": False,
            }
        ),
        workdir=repo,
    )

    result = scheduler.run()

    assert result.decision == Decision.ACCEPT
    iteration = result.iterations[0]
    executor = json.loads(Path(iteration["executor_result"]).read_text(encoding="utf-8"))
    assert executor["git_diff_present"] is False
    assert executor["no_diff_reason_source"] == "executor"
    reviewer = json.loads(Path(iteration["reviewer_decision"]).read_text(encoding="utf-8"))
    assert "verification-only" in reviewer["rationale"]


def test_checker_can_report_invalid_implementation_with_metrics(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    executor_script = scripts / "executor.py"
    executor_script.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "Path('feature.txt').write_text('slow\\n', encoding='utf-8')",
                "Path(os.environ['EAL_EXECUTOR_RESULT']).write_text(",
                "    json.dumps({'status': 'completed', 'summary': 'created slow feature'}, indent=2) + '\\n',",
                "    encoding='utf-8',",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    checker_script = scripts / "checker.py"
    checker_script.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "Path(os.environ['EAL_CHECKER_RESULT']).write_text(",
                "    json.dumps({",
                "        'benchmark_valid': True,",
                "        'implementation_valid': False,",
                "        'verdict': 'VALID_INEFFECTIVE',",
                "        'recommendation': 'NEEDS_FIX',",
                "        'rationale': 'Benchmark is clean but speed target failed.',",
                "        'speed': {'latency_ms': 99.0},",
                "        'quality': {'correctness': 'passed'},",
                "        'qualitative': {'overall': {'qualitative_pass': False, 'summary': 'visible regression'}},",
                "    }, indent=2) + '\\n',",
                "    encoding='utf-8',",
                ")",
            ]
        ),
        encoding="utf-8",
    )

    scheduler = Scheduler(
        ExperimentSpec.from_dict({"name": "slow-demo", "goal": "create feature"}),
        LabConfig.from_dict(
            {
                "executor_command": f"{sys.executable} {executor_script}",
                "checker_command": f"{sys.executable} {checker_script}",
                "max_iterations": 1,
                "rollback_on_reject": False,
            }
        ),
        workdir=repo,
    )

    result = scheduler.run()

    assert result.decision == Decision.NEEDS_FIX
    checker = json.loads(
        Path(result.iterations[0]["checker_result"]).read_text(encoding="utf-8")
    )
    assert checker["benchmark_valid"] is True
    assert checker["implementation_valid"] is False
    assert checker["speed"]["latency_ms"] == 99.0
    assert checker["qualitative"]["overall"]["summary"] == "visible regression"


def test_scheduler_detects_gpu_polluted_retest_reason(tmp_path):
    checker_result = tmp_path / "checker_result.json"
    checker_result.write_text(
        json.dumps(
            {
                "recommendation": "NEEDS_RETEST",
                "artifacts": {"gpu_preflight": {"status": "blocked"}},
            }
        ),
        encoding="utf-8",
    )

    assert _checker_gpu_retest_reason(checker_result) == "gpu_preflight_blocked"


def test_resource_hold_polls_until_gpu_is_clean(tmp_path, monkeypatch):
    probes = [
        {
            "processes": [
                {
                    "pid": 123,
                    "process_name": "external",
                    "used_gpu_memory_mib": 4096,
                }
            ]
        },
        {"processes": []},
    ]
    sleeps = []

    def fake_probe(threshold_mib):
        assert threshold_mib == 1024
        return probes.pop(0)

    monkeypatch.setattr(
        scheduler_module,
        "_gpu_compute_processes_over_threshold",
        fake_probe,
    )
    monkeypatch.setattr(scheduler_module.time, "sleep", sleeps.append)

    result = _hold_until_resources_clean(
        LabConfig.from_dict(
            {
                "executor_command": "python executor.py",
                "checker_command": "python benchmark.py",
                "resource_hold_enabled": True,
                "resource_hold_poll_seconds": 300,
                "resource_hold_timeout_seconds": 0,
            }
        ),
        EventLog(tmp_path / "events.jsonl"),
        7,
        role="Reviewer Agent",
        context="reviewer_retest",
        reason="gpu_preflight_blocked",
    )

    assert result["status"] == "idle"
    assert sleeps == [300.0]
    events = _read_jsonl(tmp_path / "events.jsonl")
    assert [event["event"] for event in events] == [
        "reviewer_retest_resource_hold_started",
        "reviewer_retest_resource_hold_poll",
        "reviewer_retest_resource_hold_finished",
    ]
    assert events[1]["role"] == "Reviewer Agent"
    assert events[1]["data"]["external_compute_processes"][0]["pid"] == 123


def test_external_reviewer_command_assigns_executor_task(tmp_path):
    repo = _init_repo(tmp_path / "repo")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    reviewer_script = scripts / "reviewer.py"
    reviewer_script.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "if os.environ['EAL_REVIEWER_CONTEXT'] == 'before_executor':",
                "    Path(os.environ['EAL_REVIEWER_NEXT_ACTION']).write_text(",
                "    json.dumps({",
                "        'target_role': 'Executor Agent',",
                "        'instructions': 'Create reviewer_task.txt with the reviewer instruction.',",
                "        'source': 'test_reviewer',",
                "    }, indent=2) + '\\n',",
                "    encoding='utf-8',",
                "    )",
                "else:",
                "    Path(os.environ['EAL_REVIEWER_DECISION']).write_text(",
                "    json.dumps({'decision': 'ACCEPT', 'rationale': 'External reviewer accepted.'}, indent=2) + '\\n',",
                "    encoding='utf-8',",
                "    )",
            ]
        ),
        encoding="utf-8",
    )
    executor_script = scripts / "executor.py"
    executor_script.write_text(
        "\n".join(
            [
                "import json, os",
                "from pathlib import Path",
                "action = json.loads(Path(os.environ['EAL_REVIEWER_NEXT_ACTION']).read_text(encoding='utf-8'))",
                "Path('reviewer_task.txt').write_text(action['instructions'] + '\\n', encoding='utf-8')",
                "Path(os.environ['EAL_EXECUTOR_RESULT']).write_text(",
                "    json.dumps({'status': 'completed', 'summary': action['instructions']}, indent=2) + '\\n',",
                "    encoding='utf-8',",
                ")",
            ]
        ),
        encoding="utf-8",
    )

    scheduler = Scheduler(
        ExperimentSpec.from_dict({"name": "reviewer-demo", "goal": "use reviewer output"}),
        LabConfig.from_dict(
            {
                "reviewer_command": f"{sys.executable} {reviewer_script}",
                "executor_command": f"{sys.executable} {executor_script}",
                "checker_command": f"{sys.executable} -c \"print('ok')\"",
                "max_iterations": 1,
                "rollback_on_reject": False,
            }
        ),
        workdir=repo,
    )

    result = scheduler.run()

    assert result.decision == Decision.ACCEPT
    action = json.loads(
        Path(result.iterations[0]["reviewer_next_action"]).read_text(encoding="utf-8")
    )
    assert action["source"] == "test_reviewer"
    assert action["command"]["exit_code"] == 0
    assert (repo / "reviewer_task.txt").read_text(encoding="utf-8").strip() == action[
        "instructions"
    ]


def test_demo_trace_does_not_require_git_or_run_commands(tmp_path):
    run_dir = run_demo(
        ExperimentSpec.from_dict({"name": "demo-trace", "goal": "show role order"}),
        LabConfig.from_dict(
            {
                "executor_command": "exit 99",
                "checker_command": "exit 99",
                "max_iterations": 1,
            }
        ),
        workdir=tmp_path,
    )

    events = _read_jsonl(run_dir / "events.jsonl")
    reviewer_next_action = json.loads(
        (run_dir / "iter-001" / "reviewer_next_action.json").read_text(encoding="utf-8")
    )
    executor_prompt = run_dir / "initial_prompts" / "executor_initial_prompt.md"

    assert [event["role"] for event in events if event["event"] == "wake"] == [
        "Reviewer Agent",
        "Executor Agent",
        "Checker Step",
        "Reviewer Agent",
    ]
    assert reviewer_next_action["target_role"] == "Executor Agent"
    assert executor_prompt.exists()
    assert "README_EXECUTOR.md" in executor_prompt.read_text(encoding="utf-8")
    assert not (tmp_path / "executor_result.json").exists()


def _init_repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init")
    _git(path, "config", "user.email", "test@example.com")
    _git(path, "config", "user.name", "Test User")
    (path / "README.md").write_text("# temp repo\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    return path


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
