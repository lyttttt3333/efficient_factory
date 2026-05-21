# efficient-agent-loop

A minimal three-role experiment loop for code optimization experiments.

This repo does not implement inference optimization logic. It provides the
scheduler skeleton that coordinates this order:

- `Reviewer Agent`: reads the current context and writes the next action.
- `Executor Agent`: runs a configured implementation command and writes
  `executor_result.json`.
- `Checker Step`: runs a configured benchmark/correctness command and writes
  `checker_result.json`.
- `Reviewer Agent`: reads executor output, git diff, and checker output, then
  writes `reviewer_decision.json`.

The first version is single-process and sequential. There is no MCP server, web
UI, or concurrency layer.

## New Machine Setup

The scheduler package is intentionally lightweight. It has no CUDA, conda, MCP,
model-weight, or web-service dependency. Those requirements belong to the
Executor or Checker commands you put in `lab.yaml`, not to the loop framework
itself.

Minimum environment:

- Python 3.10 or newer
- `git` on `PATH` for real `run` mode, because the scheduler collects git diffs
- `PyYAML` if you use `.yaml` lab files
- `pytest` only if you want to run tests

Recommended isolated setup:

```bash
git clone https://github.com/lyttttt3333/efficient_factory.git
cd efficient_factory/efficient-agent-loop

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[yaml,test]"
```

Runtime-only install options:

```bash
# JSON lab files only
python -m pip install -e .

# YAML lab files
python -m pip install -e ".[yaml]"
```

Verify the install without touching `playground/`:

```bash
efficient-agent-loop demo \
  --experiment examples/experiment.json \
  --lab examples/lab.yaml \
  --workdir .

python -m pytest -q
```

For local development without installing:

```bash
PYTHONPATH=src python -m efficient_agent_loop.cli --help
```

`codex` is optional from this package's perspective. Install and authenticate
the Codex CLI only if a lab config uses `codex exec ...` as one of the role
commands.

## Quick Start

Run inside a git worktree:

```bash
efficient-agent-loop run \
  --experiment examples/experiment.json \
  --lab examples/lab.yaml \
  --workdir .
```

Equivalent local-development command:

```bash
PYTHONPATH=src python -m efficient_agent_loop.cli run \
  --experiment examples/experiment.json \
  --lab examples/lab.yaml \
  --workdir .
```

## Lab Configuration

`examples/lab.yaml` configures the commands:

```yaml
executor_command: "..."
checker_command: "..."
reviewer_command: ""
executor_permission: "auto_review"
checker_permission: "auto_review"
reviewer_permission: "auto_review"
executor_readme: "README_EXECUTOR.md"
checker_readme: "README_CHECKER.md"
reviewer_readme: "README_REVIEWER.md"
max_iterations: 0  # no scheduler iteration limit; Reviewer ACCEPT/REJECT/STOP ends the loop
commit_on_accept: false
rollback_on_reject: true
```

If `reviewer_command` is empty, the scheduler uses deterministic built-in
Reviewer behavior:

- before Executor: write a minimal `reviewer_next_action.json` from the
  ExperimentSpec or previous Reviewer feedback
- after Checker: require an Executor-authored explanation for no-diff
  verification-only iterations, return `NEEDS_RETEST` for invalid or polluted
  benchmark results, `ACCEPT` for valid/effective results, and `NEEDS_FIX` for
  unexplained no-diff or valid but ineffective implementations

Set `max_iterations: 0` to remove the scheduler-side iteration cap. In that
mode the loop continues until Reviewer returns `ACCEPT`, `REJECT`, or `STOP`.

Permissions are role policy metadata for downstream runners. All built-in
roles default to `auto_review`: the runner should use sandboxed automatic
execution and let the approval/review layer decide whether an escalation is
allowed. Role prompts and reviewer instructions still define the intended
capability boundary: Executor edits code, Checker runs benchmarks and writes
artifacts, and Reviewer reads artifacts and decides the next action.

The scheduler does not elevate OS permissions by itself. It records the policy
in `events.jsonl` and exposes it as `EAL_AGENT_PERMISSION` so a future sandbox
runner or `codex exec` wrapper can enforce it.

## Runtime Artifacts

Each run writes artifacts under `.eal/runs/<experiment>/<timestamp>/iter-XXX/`:

- `../events.jsonl`
- `../initial_prompts/reviewer_initial_prompt.md`
- `../initial_prompts/executor_initial_prompt.md`
- `../initial_prompts/checker_initial_prompt.md`
- `reviewer_next_action.json`
- `executor_result.json`
- `git_diff.patch`
- `checker_result.json`
- `reviewer_decision.json`

The scheduler excludes `.eal/` from collected experiment diffs.

`events.jsonl` is the global trace log. Each line records:

- UTC `timestamp`
- `role`
- `event`
- human-readable `message`
- optional `iteration`
- optional structured `data`

Reviewer `output` events include the actual JSON content written by Reviewer,
so you can inspect what it asked Executor or Checker to do and what final
decision it returned.

Role order:

```text
Reviewer -> Executor -> Checker -> Reviewer
```

Example event:

```json
{"data":{"artifact":".../reviewer_next_action.json","content":{"target_role":"Executor Agent"}},"event":"output","iteration":1,"message":"Reviewer wrote the next action.","role":"Reviewer Agent","timestamp":"2026-05-19T12:00:00+00:00"}
```

## No-op Demo Trace

To see the role order without modifying code, collecting git diff, or running
benchmarks:

```bash
efficient-agent-loop demo \
  --experiment examples/experiment.json \
  --lab examples/lab.yaml \
  --workdir .
```

This writes:

```text
.eal/demo-runs/<experiment>/<timestamp>/events.jsonl
.eal/demo-runs/<experiment>/<timestamp>/demo_result.json
.eal/demo-runs/<experiment>/<timestamp>/initial_prompts/*.md
.eal/demo-runs/<experiment>/<timestamp>/iter-001/reviewer_next_action.json
```

## Command Environment

Executor, checker, and reviewer commands receive these environment variables:

- `EAL_EXPERIMENT_JSON`
- `EAL_WORKDIR`
- `EAL_RUN_DIR`
- `EAL_ITERATION`
- `EAL_EXECUTOR_RESULT`
- `EAL_CHECKER_RESULT`
- `EAL_REVIEWER_NEXT_ACTION`
- `EAL_REVIEWER_DECISION`
- `EAL_REVIEWER_CONTEXT`
- `EAL_AGENT_INITIAL_PROMPT`
- `EAL_AGENT_README`
- `EAL_REVIEWER_INITIAL_PROMPT`
- `EAL_EXECUTOR_INITIAL_PROMPT`
- `EAL_CHECKER_INITIAL_PROMPT`
- `EAL_REVIEWER_README`
- `EAL_EXECUTOR_README`
- `EAL_CHECKER_README`
- `EAL_GIT_DIFF`
- `EAL_PREVIOUS_REVIEWER_DECISION`

Each role receives an initial prompt file. The prompt tells it to read its role
README before acting:

- Reviewer -> `README_REVIEWER.md`
- Executor -> `README_EXECUTOR.md`
- Checker -> `README_CHECKER.md`

These paths can be overridden in `lab.yaml`. Missing README files are allowed;
the initial prompt records the file as missing so a generic toy repo still runs.

## Reviewer Command

`reviewer_command` is one command invoked whenever Reviewer is woken. It should
look at `EAL_REVIEWER_CONTEXT`:

- `before_executor`: write `EAL_REVIEWER_NEXT_ACTION`.
- `after_checker`: write `EAL_REVIEWER_DECISION`.

Next-action JSON should include at least:

```json
{
  "target_role": "Executor Agent",
  "instructions": "Concrete task for Executor.",
  "checker_commands": ["python benchmark.py"],
  "acceptance_criteria": ["What Checker should verify."]
}
```

Decision JSON must include:

```json
{
  "decision": "ACCEPT",
  "rationale": "Benchmark passed and diff is non-empty.",
  "next_instructions": ""
}
```

Valid decisions are `ACCEPT`, `REJECT`, `NEEDS_FIX`, `NEEDS_RETEST`, and `STOP`.

## Executor Contract

The executor command should write `EAL_EXECUTOR_RESULT`. If it does not, the
scheduler writes a fallback result. If there is no code diff, the result must
include `no_diff_reason`; the scheduler will add a fallback explanation if the
executor omitted one.

No diff is not itself a success or failure signal. It is an Executor
accountability check:

- no diff without an Executor-authored `no_diff_reason` means Executor needs to
  fix the handoff
- no diff with a concrete reason means Reviewer should continue to judge the
  experiment from Checker evidence
- accepted no-diff iterations are verification-only; if commit-on-accept is
  enabled, the scheduler skips the commit because there is no experiment diff

## Checker Contract

The checker command may run benchmarks and write benchmark artifacts. It should
not intentionally modify source code. The scheduler records whether the git diff
changed during the checker step and treats that as benchmark pollution.

Checker benchmark commands may write logs, caches, and metrics under
`EAL_RUN_DIR` or `.eal/`.

The checker can write `EAL_CHECKER_RESULT` itself. The scheduler preserves and
enriches these fields:

```json
{
  "benchmark_valid": true,
  "implementation_valid": true,
  "verdict": "VALID_EFFECTIVE",
  "recommendation": "ACCEPT",
  "rationale": "Benchmark is clean and speed/quality meet the target.",
  "speed": {"latency_ms": 12.3},
  "quality": {"correctness": "passed"},
  "qualitative": {
    "overall": {
      "qualitative_pass": true,
      "summary": "Candidate image is visually comparable to baseline."
    }
  }
}
```

`quality` is intended for quantitative or programmatic checks such as PSNR,
MSE, correctness, and threshold flags. `qualitative` is intended for direct
artifact review, such as a multimodal baseline-vs-candidate image judgment.
For visual outputs this qualitative review is required: if the artifact pair is
missing or the multimodal evaluator cannot run, Checker should request
`NEEDS_RETEST`; if visual regression is severe, Checker should request
`NEEDS_FIX`. The scheduler preserves this field and passes the resulting
recommendation to Reviewer.

If the checker pollutes the git diff, the scheduler forces
`benchmark_valid=false`, `verdict=BENCHMARK_POLLUTED`, and
`recommendation=NEEDS_RETEST`.

## Tests

```bash
PYTHONPATH=src python -m pytest
```

## Auto Deploy Flux Example

The `examples/auto_deploy/` adapter includes a Reviewer command that scans
`/home/xieenze/yitongl/auto_deploy/ComfyUI` role READMEs, active skill index,
git diff, and prior `benchmark/artifacts/*/metrics.json` files, then writes a
Flux-specific `reviewer_next_action.json`.

The Flux Reviewer is instructed to optimize skill combinations rather than
single isolated skills. It should consider cache/reuse, quantization, sparse
attention, and other compatible skills as a joint search space whose goal is
maximum end-to-end acceleration subject to benchmark correctness and required
multimodal visual quality checks.

The Flux Reviewer should not stop just because one candidate passes locally. If
the current candidate passes but compatible cache, quantization, sparse
attention, or workflow-level candidates remain unexplored, it returns
`NEEDS_FIX` with a `next_action_hint` for the next candidate. It should return
`ACCEPT` only when speed, quantitative quality, visual quality, and remaining
candidate-space checks all pass.

Reviewer-output demo:

```bash
efficient-agent-loop run \
  --experiment examples/auto_deploy/experiment_flux_sparse.json \
  --lab examples/auto_deploy/lab_flux_reviewer.yaml \
  --workdir /home/xieenze/yitongl/auto_deploy/ComfyUI
```

This lab intentionally uses a no-op executor/checker so you can inspect
Reviewer output without changing Flux code. Replace `executor_command` with a
real Executor agent command when you are ready for code edits.
