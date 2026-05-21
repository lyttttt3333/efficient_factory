from __future__ import annotations

from efficient_skill.common.workflow import Workflow, next_node_id, output_ref, replace_model_input


def delta_reuse_cache_node(
    model_ref: list,
    delta_scale: float = 1.0,
    interval: int = 2,
    warmup_steps: int = 2,
    max_skip_steps: int = 1,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    verbose: bool = False,
) -> dict:
    return {
        "class_type": "DeltaReuseCache",
        "inputs": {
            "model": model_ref,
            "delta_scale": delta_scale,
            "interval": interval,
            "warmup_steps": warmup_steps,
            "max_skip_steps": max_skip_steps,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "verbose": verbose,
        },
    }


def insert_delta_reuse_cache(
    workflow: Workflow,
    model_ref: list,
    delta_scale: float = 1.0,
    interval: int = 2,
    warmup_steps: int = 2,
    max_skip_steps: int = 1,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    verbose: bool = False,
) -> tuple[Workflow, list]:
    node_id = next_node_id(workflow)
    workflow[node_id] = delta_reuse_cache_node(
        model_ref=model_ref,
        delta_scale=delta_scale,
        interval=interval,
        warmup_steps=warmup_steps,
        max_skip_steps=max_skip_steps,
        start_percent=start_percent,
        end_percent=end_percent,
        verbose=verbose,
    )
    new_model_ref = output_ref(node_id, 0)
    replace_model_input(workflow, old_ref=model_ref, new_ref=new_model_ref)
    workflow[node_id]["inputs"]["model"] = model_ref
    return workflow, new_model_ref
