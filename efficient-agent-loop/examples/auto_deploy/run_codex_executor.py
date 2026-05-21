from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path


def main() -> int:
    workdir = Path(os.environ["EAL_WORKDIR"]).resolve()
    run_dir = Path(os.environ["EAL_RUN_DIR"]).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_log = run_dir / "executor_codex.stdout.jsonl"
    stderr_log = run_dir / "executor_codex.stderr.txt"
    prompt_path = run_dir / "executor_codex_prompt.md"
    last_message = run_dir / "executor_codex_last_message.md"

    prompt = _build_prompt()
    prompt_path.write_text(prompt, encoding="utf-8")

    started = time.monotonic()
    permission = os.environ.get("EAL_AGENT_PERMISSION", "")
    command = [
        "codex",
        "exec",
        "--json",
        "-C",
        str(workdir),
        "-o",
        str(last_message),
    ]
    if permission == "full_access":
        command.extend(["--full-auto", "--sandbox", "workspace-write"])
    else:
        command.extend(["--full-auto", "--sandbox", "workspace-write"])
    command.append("-")
    process = subprocess.Popen(
        command,
        cwd=workdir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=os.environ.copy(),
    )
    assert process.stdin is not None
    process.stdin.write(prompt)
    process.stdin.close()

    threads = [
        threading.Thread(
            target=_copy_stream,
            args=(process.stdout, stdout_log),
            daemon=True,
        ),
        threading.Thread(
            target=_copy_stream,
            args=(process.stderr, stderr_log),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    exit_code = process.wait()
    for thread in threads:
        thread.join()

    result_path = Path(os.environ["EAL_EXECUTOR_RESULT"])
    if not result_path.exists():
        summary = "Codex executor finished without writing executor_result.json."
        if last_message.exists():
            text = last_message.read_text(encoding="utf-8").strip()
            if text:
                summary = text[:4000]
        _write_json(
            result_path,
            {
                "iteration": int(os.environ.get("EAL_ITERATION", "1")),
                "status": "completed" if exit_code == 0 else "failed",
                "summary": summary,
                "no_diff_reason": None,
                "artifacts": {
                    "codex_stdout": str(stdout_log),
                    "codex_stderr": str(stderr_log),
                    "codex_prompt": str(prompt_path),
                    "codex_last_message": str(last_message),
                },
                "executor_wrapper": {
                    "command": command,
                    "exit_code": exit_code,
                    "duration_seconds": round(time.monotonic() - started, 6),
                },
            },
        )
    return exit_code


def _build_prompt() -> str:
    experiment = _read_text_env("EAL_EXPERIMENT_JSON")
    next_action = _read_text_env("EAL_REVIEWER_NEXT_ACTION")
    initial_prompt = _read_text_env("EAL_AGENT_INITIAL_PROMPT")
    readme = _read_text_env("EAL_AGENT_README")
    previous_decision = _read_text_env("EAL_PREVIOUS_REVIEWER_DECISION")
    permission = os.environ.get("EAL_AGENT_PERMISSION", "")
    return "\n\n".join(
        [
            "# efficient-agent-loop Executor Agent",
            "You are the Executor Agent. Implement the Reviewer next action in the current git worktree.",
            f"Configured executor permission: `{permission or 'unspecified'}`. Do not decide whether the experiment succeeded; Checker and Reviewer own that.",
            "Runtime permission note: this host rejects Codex dangerous full-access mode in non-interactive cloud runs. The executor is launched with the strongest currently allowed writable mode, `--full-auto --sandbox workspace-write`, so direct file edits inside the target worktree should be used before shell-based edit fallbacks.",
            "Important local naming note: this framework uses `reviewer_next_action.json`. If repo-local README files mention `reviewer_plan.json`, treat that as the same handoff concept for this run.",
            "Before editing, read the role README and follow its dirty-tree snapshot instructions. Preserve pre-existing unrelated dirty-tree changes.",
            "This non-interactive `codex exec` environment cannot service approval prompts. Do not run commands that are likely to require approval or are unavailable here, especially `rg`, `find`, `git ls-files`, command pipelines, shell redirection, `||`, `&&`, or other shell-control operators. Use simple `grep -R`, `sed`, `ls`, or a trusted `conda run -n auto_deploy_flux_eff python -c ...` helper instead.",
            "For existence checks, run a plain `ls path` and handle a non-zero result in your reasoning; do not hide failures with redirection or `|| true`.",
            "Use the normal file_change/apply_patch mechanism for authorized edits inside the current worktree. If that fails, use the already trusted command prefix `conda run -n auto_deploy_flux_eff python -c ...` only for a small deterministic fallback scoped to files explicitly authorized by the Reviewer next action.",
            "For the README's py_compile verification, avoid the `find ... | xargs ...` form. Use a trusted Python helper that lists `efficient_skill`, `draft_canvas`, and `benchmark` files with pathlib and calls py_compile on each file.",
            "Do not use this trusted edit channel for network access, sudo, permission changes, destructive cleanup, model-weight changes, or paths outside the current git worktree.",
            "When done, write JSON to the exact `EAL_EXECUTOR_RESULT` path. If you intentionally make no code changes, include a concrete `no_diff_reason`.",
            "Run only focused smoke checks that are reasonable before the benchmark. Do not run the full benchmark; Checker will do that.",
            "## Required output path",
            os.environ["EAL_EXECUTOR_RESULT"],
            "## ExperimentSpec",
            experiment,
            "## Reviewer Next Action",
            next_action,
            "## Generated Initial Prompt",
            initial_prompt,
            "## Role README",
            readme,
            "## Previous Reviewer Decision",
            previous_decision or "(none)",
        ]
    )


def _read_text_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        return ""
    path = Path(value)
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def _copy_stream(stream, path: Path) -> None:
    if stream is None:
        return
    with path.open("w", encoding="utf-8") as handle:
        for line in stream:
            handle.write(line)
            handle.flush()


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
