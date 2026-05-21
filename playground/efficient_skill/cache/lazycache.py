from __future__ import annotations

from efficient_skill.common.workflow import Workflow, next_node_id, output_ref, replace_model_input


def lazycache_node(
    model_ref: list,
    reuse_threshold: float = 0.2,
    start_percent: float = 0.15,
    end_percent: float = 0.95,
    verbose: bool = False,
) -> dict:
    return {
        "class_type": "LazyCache",
        "inputs": {
            "model": model_ref,
            "reuse_threshold": reuse_threshold,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "verbose": verbose,
        },
    }


def insert_lazycache(
    workflow: Workflow,
    model_ref: list,
    reuse_threshold: float = 0.2,
    start_percent: float = 0.15,
    end_percent: float = 0.95,
    verbose: bool = False,
) -> tuple[Workflow, list]:
    node_id = next_node_id(workflow)
    workflow[node_id] = lazycache_node(
        model_ref=model_ref,
        reuse_threshold=reuse_threshold,
        start_percent=start_percent,
        end_percent=end_percent,
        verbose=verbose,
    )
    new_model_ref = output_ref(node_id, 0)
    replace_model_input(workflow, old_ref=model_ref, new_ref=new_model_ref)
    workflow[node_id]["inputs"]["model"] = model_ref
    return workflow, new_model_ref
