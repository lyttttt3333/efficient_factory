from __future__ import annotations

from efficient_skill.common.workflow import Workflow, next_node_id, output_ref, replace_model_input


def sparse_attention_node(
    model_ref: list,
    method: str,
    apply_to: str = "single",
    min_tokens: int = 128,
    max_tokens: int = 1_000_000,
    density: float = 0.15,
    block_size: int = 64,
    topk: float = 0.25,
    verbose: bool = False,
    precompile_tokens: int = 0,
    precompile_heads: int = 24,
    precompile_head_dim: int = 128,
) -> dict:
    return {
        "class_type": "SparseAttentionModel",
        "inputs": {
            "model": model_ref,
            "method": method,
            "apply_to": apply_to,
            "min_tokens": min_tokens,
            "max_tokens": max_tokens,
            "density": density,
            "block_size": block_size,
            "topk": topk,
            "verbose": verbose,
            "precompile_tokens": precompile_tokens,
            "precompile_heads": precompile_heads,
            "precompile_head_dim": precompile_head_dim,
        },
    }


def insert_sparse_attention(
    workflow: Workflow,
    model_ref: list,
    method: str,
    **kwargs,
) -> tuple[Workflow, list]:
    node_id = next_node_id(workflow)
    workflow[node_id] = sparse_attention_node(model_ref=model_ref, method=method, **kwargs)
    new_model_ref = output_ref(node_id, 0)
    replace_model_input(workflow, old_ref=model_ref, new_ref=new_model_ref)
    workflow[node_id]["inputs"]["model"] = model_ref
    return workflow, new_model_ref
