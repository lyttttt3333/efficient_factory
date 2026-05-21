from __future__ import annotations

from efficient_skill.common.workflow import Workflow
from efficient_skill.sparse_attention.node import insert_sparse_attention, sparse_attention_node


def spargeattn_sparse_attention_node(model_ref: list, **kwargs) -> dict:
    kwargs.setdefault("topk", 0.25)
    return sparse_attention_node(model_ref=model_ref, method="spargeattn", **kwargs)


def insert_spargeattn_sparse_attention(workflow: Workflow, model_ref: list, **kwargs) -> tuple[Workflow, list]:
    kwargs.setdefault("topk", 0.25)
    return insert_sparse_attention(workflow, model_ref=model_ref, method="spargeattn", **kwargs)
