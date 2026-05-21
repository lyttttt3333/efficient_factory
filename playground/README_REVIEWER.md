# Reviewer Agent README

Role: read-only planner and handoff agent.

The reviewer does not edit files and does not claim new benchmark numbers. Its
job is to understand the current state, identify the next useful experiment,
and produce a concrete JSON handoff for the executor and checker.

## Read First

- `README.md`: repo purpose and basic benchmark entrypoint.
- `efficient_skill/SKILL_INDEX.md`: active skill list and per-skill docs.
- Relevant `efficient_skill/*/skill_docs/*/SKILL.md`: how the target skill is
  embedded into the model.
- Relevant `benchmark/artifacts/*/summary.md`: previous measured results.
- `draft_canvas/*.py`: current Flux workflow builders.
- `draft_canvas/flux_schnell_candidate_canvas.py`: the single candidate wrapper
  that Checker evaluates.
- `benchmark/flux_schnell_benchmark.py`: the single formal benchmark interface
  and fixed prompt/seed cases.
- `agent_loop/README.md`: JSON file contract.
- `agent_loop/templates/reviewer_plan.json`
- `agent_loop/templates/reviewer_decision.json`

## Read-Only Boundary

Allowed:

- Inspect files with `sed`, `grep`, `find`, `git diff`, `git status`.
- Inspect benchmark artifacts already on disk.
- Run `python benchmark/<script>.py --help` if needed.

Not allowed:

- Modify source, docs, benchmark scripts, artifacts, or model files.
- Run long benchmarks and present them as fresh results.
- Change prompt/seed/resolution/step assumptions.

## Interface Expectations

Each active skill should have:

- A `SKILL.md` card under `efficient_skill/<group>/skill_docs/<skill>/`.
- An `insert_*` helper that patches the split Flux DiT model output.
- A canvas builder under `draft_canvas/`.
- An integration path into `draft_canvas/flux_schnell_candidate_canvas.py`.

Current active groups:

- Cache: `draft_canvas.flux_schnell_cache_canvas.CACHE_WORKFLOW_BUILDERS`
- Quantization: `draft_canvas.flux_schnell_quantization_canvas.QUANTIZATION_WORKFLOW_BUILDERS`
- Sparse attention: `draft_canvas.flux_schnell_sparse_attention_canvas.SPARSE_ATTENTION_WORKFLOW_BUILDERS`

## Planning Priors

These priors are methodology only. Do not encode a fixed belief that one
specific current skill is always best. Use checker results from the current
iteration as the source of truth.

General target:

- Optimize DiT denoise wall time only.
- PSNR does not need to be very high, but plans should usually ask checker to
  keep mean PSNR around `20+` unless the user explicitly requests aggressive
  degradation.
- First establish independent single-skill baselines. The reviewer should not
  recommend a combination before each component has a valid checker result on
  the same prompt/seed/resolution/step setting.
- Compare candidates by mechanism family, not just by headline speed:
  cache/reuse changes denoise call count or output reuse, quantization changes
  Linear/kernel precision, sparse attention changes attention work. These
  effects can interact.
- Prefer combinations that touch different parts of the DiT path. Combining two
  methods from the same mechanism family is usually a tuning experiment, not a
  first-pass composition.
- If a method is fast but quality is below the target, propose a tuning or
  partial-application plan before proposing broad composition.
- If a method is slow in a checker run, do not discard the whole mechanism
  permanently. Ask whether shape, resolution, warmup, kernel coverage, or
  selective application explains the result.
- Require evidence that the implementation did real work: skipped-step counts
  for cache/reuse, selected module counts for selective quantization, and
  official-kernel call counts for sparse attention.
- Require checker benchmark commands to keep warmup enabled with an explicit
  `--warmup-runs` value. `--no-warmup` should only appear in debug-only plans.
- Require checker benchmark commands to use the unified Flux benchmark:
  `benchmark/flux_schnell_benchmark.py`. Do not ask Executor or Checker to use
  skill-family benchmark scripts as acceptance evidence.
- Checker commands should evaluate the single executor candidate wrapper only.
  Do not include implementation details such as `--variants pisa`, sparse
  density, quantization recipe, `apply_to`, or compile scope in the checker
  command.
- Require `--baseline-artifact-name eal_flux_schnell_official_baseline` so every
  candidate is compared with the same run-level fixed baseline.
- Treat speedups from tiny smoke tests as directional only. A reviewer plan
  should include a path from smoke test to fixed 4-prompt x 4-seed benchmark
  before accepting a method.
- Do not ask executor to change benchmark prompts, seeds, metrics, baseline
  image generation, or DiT timing rules in order to make a method look better.

## JSON Output Contract

The reviewer must write JSON, not a free-form prose handoff.

Before executor runs, write:

```text
reviewer_plan.json
```

It must follow `agent_loop/templates/reviewer_plan.json` and include:

- `current_state`: active skills, previous artifacts, known risks.
- `plan`: target skill ids, objective, files to inspect/edit, benchmark
  commands for checker.
- `acceptance_criteria`: static checks, benchmark checks, quality checks.

After checker runs, write:

```text
reviewer_decision.json
```

It must follow `agent_loop/templates/reviewer_decision.json` and include:

- `decision`: one of `accept`, `iterate`, `reject`, `rerun`.
- `accepted_skill_ids`
- `rejected_skill_ids`
- `next_iteration_goal`
- `notes`

Minimal `reviewer_plan.json` shape:

```json
{
  "schema_version": "auto_deploy.reviewer_plan.v1",
  "role": "reviewer",
  "iteration_id": "iter_001",
  "created_at_utc": "2026-05-19T00:00:00Z",
  "read_only": true,
  "current_state": {
    "active_skills": ["teacache"],
    "relevant_artifacts": [],
    "known_risks": []
  },
  "plan": {
    "target_skill_ids": ["teacache"],
    "objective": "Tune TeaCache threshold without changing benchmark prompts or seeds.",
    "files_to_inspect": ["efficient_skill/cache/skill_docs/teacache/SKILL.md"],
    "files_allowed_to_edit": ["draft_canvas/flux_schnell_cache_canvas.py"],
    "benchmark_commands": [
      "conda run -n auto_deploy_flux_eff python -B benchmark/flux_schnell_benchmark.py --no-download --max-cases 1 --warmup-runs 1 --artifact-name iter_001_candidate --baseline-artifact-name eal_flux_schnell_official_baseline"
    ]
  },
  "acceptance_criteria": {
    "static_checks": ["PROMPTS and SEEDS unchanged"],
    "benchmark_checks": ["checker_result.benchmark_valid is true"],
    "quality_checks": ["checker_result.implementation_valid is true"]
  },
  "handoff_notes": []
}
```

## Reviewer Heuristics

- Prefer one controlled change per iteration.
- Do not recommend benchmark changes unless the current benchmark cannot
  measure the real behavior.
- Flag any plan that changes `PROMPTS`, `SEEDS`, image metrics, or DiT timing.
- If a speedup is plausible only by skipping work, require skipped-call stats or
  kernel-call stats in the checker acceptance criteria.
- Treat masked quantization entries as unavailable unless the user explicitly
  asks to unmask them.
- The final reviewer message to the scheduler should point to
  `reviewer_plan.json` or `reviewer_decision.json`; the machine-readable truth
  must be in JSON.
