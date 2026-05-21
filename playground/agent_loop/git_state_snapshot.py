from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "auto_deploy.git_state_snapshot.v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_git(args: list[str], repo_root: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=repo_root).decode("utf-8", errors="replace")


def _run_git_bytes(args: list[str], repo_root: Path) -> bytes:
    return subprocess.check_output(["git", *args], cwd=repo_root)


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_status_z(raw: bytes) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    parts = raw.split(b"\0")
    index = 0
    while index < len(parts):
        item = parts[index]
        index += 1
        if not item:
            continue
        text = item.decode("utf-8", errors="surrogateescape")
        status = text[:2]
        path = text[3:] if len(text) > 3 else ""
        old_path = None
        if status[0] in {"R", "C"} or status[1] in {"R", "C"}:
            if index < len(parts) and parts[index]:
                old_path = parts[index].decode("utf-8", errors="surrogateescape")
                index += 1
        entries[path] = {
            "status": status,
            "path": path,
            "old_path": old_path,
        }
    return entries


def collect_state(repo_root: Path) -> dict[str, Any]:
    raw_status = _run_git_bytes(["status", "--porcelain=v1", "-z", "--untracked-files=all"], repo_root)
    entries = _parse_status_z(raw_status)
    for path, entry in entries.items():
        full_path = repo_root / path
        entry["exists"] = full_path.exists()
        entry["sha256"] = _sha256(full_path)
        entry["size"] = full_path.stat().st_size if full_path.exists() and full_path.is_file() else None
    head = _run_git(["rev-parse", "HEAD"], repo_root).strip()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _now(),
        "repo_root": str(repo_root),
        "head": head,
        "dirty_count": len(entries),
        "entries": entries,
    }


def diff_states(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    before = baseline.get("entries", {})
    after = current.get("entries", {})
    paths = sorted(set(before) | set(after))
    changed: list[dict[str, Any]] = []
    unchanged_preexisting: list[str] = []
    resolved_preexisting: list[str] = []
    for path in paths:
        old = before.get(path)
        new = after.get(path)
        if old is None and new is not None:
            changed.append({"path": path, "change_type": "new_dirty", "before": None, "after": new})
            continue
        if old is not None and new is None:
            resolved_preexisting.append(path)
            changed.append({"path": path, "change_type": "resolved_or_removed_dirty", "before": old, "after": None})
            continue
        if old is None or new is None:
            continue
        old_sig = (old.get("status"), old.get("sha256"), old.get("exists"), old.get("size"))
        new_sig = (new.get("status"), new.get("sha256"), new.get("exists"), new.get("size"))
        if old_sig != new_sig:
            changed.append({"path": path, "change_type": "modified_since_snapshot", "before": old, "after": new})
        else:
            unchanged_preexisting.append(path)
    return {
        "schema_version": "auto_deploy.git_state_diff.v1",
        "created_at_utc": _now(),
        "repo_root": current.get("repo_root"),
        "baseline_created_at_utc": baseline.get("created_at_utc"),
        "baseline_head": baseline.get("head"),
        "current_head": current.get("head"),
        "preexisting_dirty_count": len(before),
        "current_dirty_count": len(after),
        "changed_since_snapshot_count": len(changed),
        "unchanged_preexisting_dirty_count": len(unchanged_preexisting),
        "resolved_preexisting_dirty_count": len(resolved_preexisting),
        "changed_since_snapshot": changed,
        "unchanged_preexisting_dirty_paths": unchanged_preexisting,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    snap = subparsers.add_parser("snapshot")
    snap.add_argument("--repo-root", type=Path, default=Path.cwd())
    snap.add_argument("--output", type=Path, default=Path("agent_loop_state/pre_run_git_state.json"))

    diff = subparsers.add_parser("diff")
    diff.add_argument("--repo-root", type=Path, default=Path.cwd())
    diff.add_argument("--baseline", type=Path, required=True)
    diff.add_argument("--output", type=Path, default=Path("agent_loop_state/executor_diff.json"))

    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    if args.command == "snapshot":
        state = collect_state(repo_root)
        output = args.output.resolve()
        _write_json(output, state)
        print(output)
    elif args.command == "diff":
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        current = collect_state(repo_root)
        report = diff_states(baseline, current)
        output = args.output.resolve()
        _write_json(output, report)
        print(output)


if __name__ == "__main__":
    main()
