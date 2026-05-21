# efficient_factory

This repository contains a minimal agent-loop framework and a clean Flux/ComfyUI
playground for optimization experiments.

## Layout

- `efficient-agent-loop/`: the lightweight Reviewer -> Executor -> Checker ->
  Reviewer scheduler and auto-deploy adapters.
- `playground/`: the ComfyUI/Flux experiment workspace used by the scheduler.


## Agent Loop Setup On A New Machine

The agent-loop framework itself is lightweight. It does not require CUDA,
conda, model weights, MCP, or a web UI. Those are only needed by whatever
Executor or Checker command you configure later.

Minimum requirements for `efficient-agent-loop/`:

- Python 3.10 or newer
- `git` on `PATH` for real experiment runs, because the scheduler records git
  diffs
- `PyYAML` only if you want to load `.yaml` lab files
- `pytest` only for running the test suite

Recommended setup:

```bash
git clone https://github.com/lyttttt3333/efficient_factory.git
cd efficient_factory/efficient-agent-loop

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[yaml,test]"

efficient-agent-loop demo \
  --experiment examples/experiment.json \
  --lab examples/lab.yaml \
  --workdir .

python -m pytest -q
```

For runtime-only use, `python -m pip install -e ".[yaml]"` is enough. If you
only use JSON lab files, even `python -m pip install -e .` is enough.

`codex` is not a Python dependency of this package. Install and authenticate the
Codex CLI separately only when your lab config uses a command such as
`codex exec ...` for Executor or Reviewer.

Large model weights, benchmark artifacts, runtime outputs, and agent run logs are
kept out of git by the top-level `.gitignore`.
