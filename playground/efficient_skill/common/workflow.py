from __future__ import annotations

from copy import deepcopy
from typing import Any


Workflow = dict[str, dict[str, Any]]
Ref = list[Any]


def clone_workflow(workflow: Workflow) -> Workflow:
    return deepcopy(workflow)


def next_node_id(workflow: Workflow) -> str:
    if not workflow:
        return "1"
    return str(max(int(node_id) for node_id in workflow) + 1)


def output_ref(node_id: str | int, output_index: int = 0) -> Ref:
    return [str(node_id), output_index]


def replace_model_input(workflow: Workflow, old_ref: Ref, new_ref: Ref) -> Workflow:
    """Replace direct MODEL inputs matching old_ref.

    This deliberately touches only inputs named "model" so skill patches remain
    local and predictable.
    """
    for node in workflow.values():
        inputs = node.get("inputs", {})
        if inputs.get("model") == old_ref:
            inputs["model"] = list(new_ref)
    return workflow

