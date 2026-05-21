from __future__ import annotations

from efficient_skill.common.workflow import Workflow
from efficient_skill.sparse_attention.node import insert_sparse_attention, sparse_attention_node


def pisa_sparse_attention_node(model_ref: list, **kwargs) -> dict:
    kwargs.setdefault("density", 0.15)
    kwargs.setdefault("block_size", 128)
    kwargs.setdefault("precompile_tokens", 1280)
    kwargs.setdefault("precompile_heads", 24)
    kwargs.setdefault("precompile_head_dim", 128)
    return sparse_attention_node(model_ref=model_ref, method="pisa", **kwargs)


def insert_pisa_sparse_attention(workflow: Workflow, model_ref: list, **kwargs) -> tuple[Workflow, list]:
    kwargs.setdefault("density", 0.15)
    kwargs.setdefault("block_size", 128)
    kwargs.setdefault("precompile_tokens", 1280)
    kwargs.setdefault("precompile_heads", 24)
    kwargs.setdefault("precompile_head_dim", 128)
    return insert_sparse_attention(workflow, model_ref=model_ref, method="pisa", **kwargs)
