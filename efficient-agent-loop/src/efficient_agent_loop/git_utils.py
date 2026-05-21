from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(RuntimeError):
    """Raised when a required git command fails."""


def run_git(
    args: list[str],
    cwd: str | Path,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=Path(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed with exit code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result


def ensure_git_repo(cwd: str | Path) -> Path:
    cwd = Path(cwd).resolve()
    result = run_git(["rev-parse", "--show-toplevel"], cwd, check=True)
    return Path(result.stdout.strip()).resolve()


def current_head(cwd: str | Path) -> str:
    result = run_git(["rev-parse", "HEAD"], cwd, check=True)
    return result.stdout.strip()


def collect_diff(
    cwd: str | Path,
    *,
    exclude_prefixes: tuple[str, ...] = (".eal/", ".codex"),
) -> str:
    cwd = Path(cwd)
    parts: list[str] = []

    unstaged = run_git(
        ["diff", "--no-ext-diff", "--", ".", ":(exclude).eal"],
        cwd,
        check=True,
    ).stdout
    staged = run_git(
        ["diff", "--cached", "--no-ext-diff", "--", ".", ":(exclude).eal"],
        cwd,
        check=True,
    ).stdout
    if unstaged:
        parts.append(unstaged)
    if staged:
        parts.append(staged)

    untracked = run_git(
        ["ls-files", "--others", "--exclude-standard"],
        cwd,
        check=True,
    ).stdout.splitlines()
    for rel_path in untracked:
        normalized = rel_path.replace("\\", "/")
        if any(normalized == prefix.rstrip("/") or normalized.startswith(prefix) for prefix in exclude_prefixes):
            continue
        patch = _diff_untracked_file(cwd, rel_path)
        if patch:
            parts.append(patch)
    return "\n".join(part.rstrip() for part in parts if part.strip()) + ("\n" if parts else "")


def commit_all(cwd: str | Path, message: str) -> str:
    run_git(["add", "-A", "--", ".", ":(exclude).eal"], cwd, check=True)
    if not has_staged_changes(cwd):
        return ""
    run_git(["commit", "-m", message], cwd, check=True)
    return current_head(cwd)


def has_staged_changes(cwd: str | Path) -> bool:
    result = run_git(["diff", "--cached", "--quiet"], cwd, check=False)
    return result.returncode == 1


def rollback_worktree(cwd: str | Path) -> None:
    run_git(["reset", "--hard", "HEAD"], cwd, check=True)
    run_git(["clean", "-fd", "-e", ".eal/"], cwd, check=True)


def _diff_untracked_file(cwd: Path, rel_path: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--no-index", "--", "/dev/null", rel_path],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise GitError(
            f"git diff --no-index failed for {rel_path}: {result.stderr.strip()}"
        )
    return result.stdout
