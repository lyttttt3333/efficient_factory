# efficient_factory

This repository contains a minimal agent-loop framework and a clean Flux/ComfyUI
playground for optimization experiments.

## Layout

- `efficient-agent-loop/`: the lightweight Reviewer -> Executor -> Checker ->
  Reviewer scheduler and auto-deploy adapters.
- `playground/`: the ComfyUI/Flux experiment workspace used by the scheduler.

Large model weights, benchmark artifacts, runtime outputs, and agent run logs are
kept out of git by the top-level `.gitignore`.
