from __future__ import annotations

from efficient_skill.common.workflow import Workflow, next_node_id, output_ref, replace_model_input


def teacache_node(
    model_ref: list,
    rel_l1_thresh: float = 0.4,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    max_skip_steps: int = 2,
    cache_device: str = "default",
    verbose: bool = False,
) -> dict:
    return {
        "class_type": "TeaCache",
        "inputs": {
            "model": model_ref,
            "rel_l1_thresh": rel_l1_thresh,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "max_skip_steps": max_skip_steps,
            "cache_device": cache_device,
            "verbose": verbose,
        },
    }


def insert_teacache(
    workflow: Workflow,
    model_ref: list,
    rel_l1_thresh: float = 0.4,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    max_skip_steps: int = 2,
    cache_device: str = "default",
    verbose: bool = False,
) -> tuple[Workflow, list]:
    node_id = next_node_id(workflow)
    workflow[node_id] = teacache_node(
        model_ref=model_ref,
        rel_l1_thresh=rel_l1_thresh,
        start_percent=start_percent,
        end_percent=end_percent,
        max_skip_steps=max_skip_steps,
        cache_device=cache_device,
        verbose=verbose,
    )
    new_model_ref = output_ref(node_id, 0)
    replace_model_input(workflow, old_ref=model_ref, new_ref=new_model_ref)
    workflow[node_id]["inputs"]["model"] = model_ref
    return workflow, new_model_ref
