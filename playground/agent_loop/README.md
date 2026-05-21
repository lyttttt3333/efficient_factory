# Agent Loop JSON Interface

This directory contains the machine-readable contract used by the
reviewer/executor/checker loop.

Required files per iteration:

- `reviewer_plan.json`: produced by the reviewer before execution.
- `executor_result.json`: produced by the executor after edits and smoke checks.
- `checker_result.json`: produced by the checker after static audit and
  benchmark conversion.
- `reviewer_decision.json`: produced by the reviewer after reading checker
  output.

Use `templates/*.json` as the minimal schema examples.

## Checker Result Wrapper

Convert a benchmark artifact into `checker_result.json`:

```bash
conda run -n auto_deploy_flux_eff python -B agent_loop/metrics_to_checker_result.py \
  --metrics benchmark/artifacts/<artifact-name>/metrics.json \
  --output benchmark/artifacts/<artifact-name>/checker_result.json \
  --static-audit-status pass
```

The wrapper computes:

- `benchmark_valid`
- `implementation_valid`
- `recommendation`
- warmup metadata validation; acceptance runs require warmup unless
  `--allow-no-warmup` is passed
- per-variant speed/quality summaries
- cache skip-step plausibility checks
- quantization masked-skill and selective-shape checks
- sparse-attention official-kernel checks
- static audit status propagation into `implementation_valid`

## Dirty Tree Snapshot

Before executor edits, capture the current dirty tree:

```bash
conda run -n auto_deploy_flux_eff python -B agent_loop/git_state_snapshot.py snapshot \
  --output agent_loop_state/pre_executor_git_state.json
```

After executor edits, compute only changes since that snapshot:

```bash
conda run -n auto_deploy_flux_eff python -B agent_loop/git_state_snapshot.py diff \
  --baseline agent_loop_state/pre_executor_git_state.json \
  --output agent_loop_state/executor_diff.json
```

The scheduler should use `changed_since_snapshot` from
`agent_loop_state/executor_diff.json` instead of raw `git status`.
