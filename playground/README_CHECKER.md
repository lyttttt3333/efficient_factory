# Checker Agent README

Role: benchmark integrity auditor and benchmark runner.

The checker verifies that the benchmark was not polluted and that an alleged
speedup comes from a real skill implementation rather than a benchmark hack.
Checker may generate benchmark artifacts and `checker_result.json`. It should
not edit source code.

## Read First

- Executor handoff and `git diff`.
- `README_EXECUTOR.md`: expected implementation contract.
- `efficient_skill/SKILL_INDEX.md`: active skill docs.
- Unified Flux benchmark script under `benchmark/flux_schnell_benchmark.py`.
- `agent_loop/README.md`
- `agent_loop/templates/checker_result.json`

## Canonical Benchmark Interface

There is one formal Flux benchmark entrypoint for every active candidate:

```text
benchmark/flux_schnell_benchmark.py
```

The checker does not select cache, quantization, sparse attention, compile, or
any other implementation detail. Executor must hide the modified model behind:

```text
draft_canvas/flux_schnell_candidate_canvas.py
```

The benchmark compares fixed official baseline vs that single candidate
wrapper. Do not route acceptance checks through skill-family benchmark scripts,
and do not pass skill-specific flags to the checker command.

The unified benchmark supports:

```bash
--max-cases
--width
--height
--steps
--guidance-scale
--max-sequence-length
--no-download
--no-warmup
--warmup-runs
--artifact-name
--baseline-artifact-name
```

Artifacts:

- `benchmark/artifacts/<artifact-name>/summary.md`
- `benchmark/artifacts/<artifact-name>/metrics.json`
- `benchmark/artifacts/<artifact-name>/metrics.csv`
- `benchmark/artifacts/<artifact-name>/outputs/`

Timing contract:

- Speedup is based on DiT denoise wall time only.
- Text encoder, VAE decode, image save, model load, quantization conversion,
  TensorRT export/build, and compile overhead are not counted as DiT speedup.
- Acceptance runs must keep warmup enabled with `--warmup-runs 1` or more.
  `--no-warmup` is only for debugging and is rejected by the checker wrapper
  unless `--allow-no-warmup` is passed explicitly.

## Static Integrity Audit

Inspect changed files before running benchmarks:

```bash
conda run -n auto_deploy_flux_eff python -B agent_loop/git_state_snapshot.py diff \
  --baseline agent_loop_state/pre_executor_git_state.json \
  --output agent_loop_state/executor_diff.json
```

Use `agent_loop_state/executor_diff.json` for executor changes. Do not treat
pre-existing dirty-tree files as executor edits.

For source review, inspect the changed paths from `changed_since_snapshot` plus
the executor's declared `changed_files`.

Reject or flag the run if the diff does any of these without explicit user
approval:

- Changes fixed `PROMPTS` or `SEEDS`.
- Changes baseline workflow to include a skill.
- Changes image metric implementation to hide quality loss.
- Reuses baseline image path as variant image path.
- Hard-codes prompt text, seed, variant name, or artifact name inside a skill.
- Fakes timing, skipped-step counts, selected-module counts, or kernel-call
  counts.
- Adds sleeps, artificial CUDA work, or forced synchronization only to the
  baseline path.
- Measures end-to-end time while reporting it as DiT-only time.
- Revives masked quantization paths without user instruction.

Useful grep probes:

```bash
grep -RIn "PROMPTS\\|SEEDS\\|compare_images\\|denoise_wall_time_s\\|baseline_image\\|variant_image" benchmark efficient_skill draft_canvas comfy_extras
grep -RIn "sleep\\|time.perf_counter\\|cuda.Event\\|synchronize\\|artifact_name\\|noise_seed" benchmark efficient_skill draft_canvas comfy_extras
```

## Smoke Benchmark Commands

Use unique artifact names so previous results are not overwritten.

Formal candidate eval example:

```bash
conda run -n auto_deploy_flux_eff python -B benchmark/flux_schnell_benchmark.py \
  --no-download --max-cases 1 --warmup-runs 1 \
  --artifact-name checker_candidate_smoke \
  --baseline-artifact-name eal_flux_schnell_official_baseline
```

Full fixed benchmark uses the same command with `--max-cases 16`. The fixed
suite is 4 prompts x 4 seeds. The benchmark default is one case for fast loop
iteration.

## Result Checks And JSON Conversion

Open `metrics.json` and `summary.md` after each run.

Required:

- `summary.num_cases` equals the requested case count.
- Every row has matching prompt/seed between baseline and variant.
- `baseline_image` and `variant_image` paths exist and are not identical.
- `baseline_dit_wall_time_s` and `variant_dit_wall_time_s` are positive.
- MSE, MAE, and PSNR are present.
- `dit_speedup` equals baseline DiT time divided by variant DiT time within
  normal floating point tolerance.
- `summary.benchmark_config.warmup_enabled` is true.
- `summary.benchmark_config.warmup_runs` is at least 1.
- `summary.benchmark_config.warmup_scope` is
  `before_each_measured_workflow`.
- Every row's warmup metadata matches `summary.benchmark_config`.

After the benchmark, convert the artifact metrics to checker JSON:

```bash
conda run -n auto_deploy_flux_eff python -B agent_loop/metrics_to_checker_result.py \
  --metrics benchmark/artifacts/<artifact-name>/metrics.json \
  --output benchmark/artifacts/<artifact-name>/checker_result.json \
  --diff-report agent_loop_state/executor_diff.json \
  --static-audit-status pass
```

The wrapper computes:

- `benchmark_valid`
- `implementation_valid`
- `recommendation`
- per-variant speed and quality summaries
- implementation plausibility checks that are visible from generic candidate
  metrics
- `implementation_valid=false` if `--static-audit-status fail`
- `benchmark_valid=false` if required warmup metadata is missing or disabled

Checker should not reject or accept based on whether a named skill appears to
have run. It should only judge benchmark validity, speed, quantitative quality,
and required visual quality for the candidate artifact.

## JSON Output Contract

The checker must write:

```text
checker_result.json
```

Normally this file should be the wrapper output at:

```text
benchmark/artifacts/<artifact-name>/checker_result.json
```

If multiple benchmark commands are run, either write one checker result per
artifact or create a top-level `checker_result.json` that references each
artifact-level result.

Minimal fields:

```json
{
  "schema_version": "auto_deploy.checker_result.v1",
  "role": "checker",
  "benchmark_valid": true,
  "implementation_valid": true,
  "recommendation": "accept",
  "static_audit": {
    "status": "pass",
    "notes": []
  },
  "integrity_checks": {
    "passed": [],
    "failed": []
  },
  "implementation_issues": [],
  "summary": {
    "num_cases": 1,
    "num_rows": 1,
    "variants": ["teacache"]
  },
  "variant_results": [],
  "next_reviewer_notes": []
}
```

The final checker message to the scheduler can summarize the result, but the
machine-readable truth must be `checker_result.json`.
