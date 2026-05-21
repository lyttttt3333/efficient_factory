from __future__ import annotations

from efficient_skill.common.workflow import Workflow, next_node_id, output_ref, replace_model_input


def torch_compile_node(model_ref: list, backend: str = "inductor") -> dict:
    return {
        "class_type": "TorchCompileModel",
        "inputs": {
            "model": model_ref,
            "backend": backend,
        },
    }


def insert_torch_compile(workflow: Workflow, model_ref: list, backend: str = "inductor") -> tuple[Workflow, list]:
    node_id = next_node_id(workflow)
    workflow[node_id] = torch_compile_node(model_ref=model_ref, backend=backend)
    new_model_ref = output_ref(node_id, 0)
    replace_model_input(workflow, old_ref=model_ref, new_ref=new_model_ref)
    workflow[node_id]["inputs"]["model"] = model_ref
    return workflow, new_model_ref
