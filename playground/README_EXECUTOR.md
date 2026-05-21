# Executor Agent README

Role: implement the reviewer plan without corrupting the benchmark.

The executor edits code and docs. It may run smoke tests, but it should not be
the final authority on benchmark validity. The checker owns final benchmark
verification. The executor result must be JSON.

## Read First

- `README_REVIEWER.md`: understand the previous handoff shape.
- `efficient_skill/SKILL_INDEX.md`: active skill inventory.
- Target skill card under `efficient_skill/<group>/skill_docs/<skill>/SKILL.md`.
- Relevant canvas in `draft_canvas/`.
- Canonical candidate wrapper: `draft_canvas/flux_schnell_candidate_canvas.py`.
- Unified Flux benchmark script in `benchmark/flux_schnell_benchmark.py`.
- `agent_loop/README.md`
- `agent_loop/templates/executor_result.json`

## Allowed Edit Areas

Normal skill work:

- `efficient_skill/`: independent skill helper functions and skill docs.
- `draft_canvas/`: Flux prototype canvas builders and active registries.
- `comfy_extras/nodes_*.py`: runtime node implementation when a skill needs a
  real model patch, attention override, or quantization wrapper.

Benchmark interface work:

- `benchmark/flux_schnell_benchmark.py` is the only formal Flux benchmark
  interface. Do not add or route checker runs through skill-family benchmark
  scripts.
- Benchmark edits are restricted. Prefer registering a new model wrapper/canvas
  variant and testing it through the unified benchmark.
- Root README files only when the agent contract changes.

Avoid:

- Changing model weights, prompt seeds, output metric formulas, or baseline
  generation.
- Editing unrelated ComfyUI/frontend/API code.
- Reviving masked quantization paths unless the user explicitly requests it.

## Dirty Tree Boundary

This repo can start with a dirty git tree. Do not report raw `git status` as the
executor diff. Use the snapshot tool so pre-existing changes are not counted as
executor work.

Before edits:

```bash
conda run -n auto_deploy_flux_eff python -B agent_loop/git_state_snapshot.py snapshot \
  --output agent_loop_state/pre_executor_git_state.json
```

After edits:

```bash
conda run -n auto_deploy_flux_eff python -B agent_loop/git_state_snapshot.py diff \
  --baseline agent_loop_state/pre_executor_git_state.json \
  --output agent_loop_state/executor_diff.json
```

Put `agent_loop_state/executor_diff.json` in `executor_result.json`.

## Skill Interface Contract

Every active skill helper should follow this shape:

```python
def insert_some_skill(workflow: Workflow, model_ref: list, ...) -> tuple[Workflow, list]:
    node_id = next_node_id(workflow)
    workflow[node_id] = some_skill_node(model_ref=model_ref, ...)
    new_model_ref = output_ref(node_id, 0)
    replace_model_input(workflow, old_ref=model_ref, new_ref=new_model_ref)
    workflow[node_id]["inputs"]["model"] = model_ref
    return workflow, new_model_ref
```

The skill should patch the split Flux DiT model output, currently
`output_ref("1", 0)`, unless a reviewer handoff explicitly says otherwise.
Text encoders and VAE are outside the default efficiency target.

## Canvas Contract

Candidate wrapper contract:

- The formal deliverable is `build_flux_schnell_candidate(...)` in
  `draft_canvas/flux_schnell_candidate_canvas.py`.
- Start from `build_flux_schnell_baseline(...)`, then wrap/patch the model with
  the current best composed optimization stack.
- Keep prompt, seed, width, height, steps, guidance, sampler, and scheduler
  parameters explicit.
- `prepare_candidate(...)` may do one-time setup that should not be counted as
  DiT denoise time.
- `clear_candidate_state()` should release any candidate-specific caches or
  global state.
- Update `CANDIDATE_METADATA` with a human-readable stack description and
  parameter values for Reviewer context.
- Checker must not need to know whether the candidate uses PISA, cache,
  quantization, compile, or a combination.

## Benchmark Contract

Do not make benchmark shortcuts. In particular:

- Do not change `PROMPTS` or `SEEDS` unless explicitly requested.
- Do not use variant images as baseline images.
- Do not fake or overwrite `denoise_wall_time_s`.
- Do not include text encoder, VAE decode, image save, quantization build, or
  model load in DiT speedup.
- Do not special-case benchmark prompt text, seed, variant name, or artifact
  path inside a skill.
- Do not alter `compare_images` to make quality look better.

The unified benchmark should expose:

- `--max-cases`
- `--width`
- `--height`
- `--steps`
- `--guidance-scale`
- `--max-sequence-length`
- `--no-download`
- `--no-warmup`
- `--warmup-runs`
- `--artifact-name`
- `--baseline-artifact-name`

Artifacts should be written under `benchmark/artifacts/<artifact-name>/` with
`summary.md`, `metrics.json`, `metrics.csv`, and `outputs/` when rows exist.
Checker handoff runs should keep warmup enabled and pass an explicit
`--warmup-runs 1` or higher. Use `--no-warmup` only for local debugging.

## Executor Verification

Before handing off to checker, run at least:

```bash
find efficient_skill draft_canvas benchmark -name '*.py' -print0 | \
  xargs -0 conda run -n auto_deploy_flux_eff python -B -m py_compile
```

Also run the unified benchmark `--help` command:

```bash
conda run -n auto_deploy_flux_eff python -B benchmark/flux_schnell_benchmark.py --help
```

## JSON Output Contract

The executor must write:

```text
executor_result.json
```

It must follow `agent_loop/templates/executor_result.json`.

Minimal shape:

```json
{
  "schema_version": "auto_deploy.executor_result.v1",
  "role": "executor",
  "iteration_id": "iter_001",
  "created_at_utc": "2026-05-19T00:00:00Z",
  "reviewer_plan_path": "reviewer_plan.json",
  "pre_executor_git_state_path": "agent_loop_state/pre_executor_git_state.json",
  "executor_diff_path": "agent_loop_state/executor_diff.json",
  "changed_files": ["draft_canvas/flux_schnell_cache_canvas.py"],
  "changed_skill_ids": ["teacache"],
  "benchmark_interface_changes": [],
  "commands_run": [],
  "smoke_results": [],
  "checker_commands": [
    "conda run -n auto_deploy_flux_eff python -B benchmark/flux_schnell_benchmark.py --no-download --max-cases 1 --warmup-runs 1 --artifact-name iter_001_candidate --baseline-artifact-name eal_flux_schnell_official_baseline"
  ],
  "known_risks": [],
  "status": "ready_for_checker"
}
```

The final executor message to the scheduler can be short, but the authoritative
handoff must be `executor_result.json`.
