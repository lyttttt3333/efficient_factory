#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

if ! git rev-parse --verify initial-state >/dev/null 2>&1; then
  echo "missing required git tag: initial-state" >&2
  exit 1
fi

git reset --hard initial-state
git clean -fd
rm -rf benchmark/artifacts agent_loop_state .eal output input temp user

status="$(git status --short)"
if [[ -n "$status" ]]; then
  echo "$status"
  echo "workspace is not clean after reset" >&2
  exit 1
fi

echo "reset to initial-state: $(git rev-parse --short HEAD)"
