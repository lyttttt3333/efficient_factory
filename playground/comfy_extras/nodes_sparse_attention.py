from __future__ import annotations

import logging

import comfy.model_patcher
from comfy_api.latest import ComfyExtension, io

from efficient_skill.sparse_attention.core import SparseAttentionConfig, SparseAttentionHolder


SPARSE_ATTENTION_METHODS = ["pisa", "spargeattn"]


def sparse_attention_sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    orig_model_options = guider.model_options
    holder = None
    try:
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        transformer_options = guider.model_options.setdefault("transformer_options", {})
        original_holder = transformer_options["sparse_attention"]
        holder = original_holder.clone()
        holder.previous_override = transformer_options.get("optimized_attention_override")
        transformer_options["sparse_attention"] = holder
        transformer_options["optimized_attention_override"] = holder.attention_override
        logging.info(
            "SparseAttention official kernel enabled - method: %s, apply_to: %s, min_tokens: %d, max_tokens: %d",
            holder.config.method,
            holder.config.apply_to,
            holder.config.min_tokens,
            holder.config.max_tokens,
        )
        return executor(*args, **kwargs)
    finally:
        if holder is not None:
            original_holder = orig_model_options.get("transformer_options", {}).get("sparse_attention")
            if original_holder is not None:
                original_holder.copy_stats_from(holder)
            logging.info("SparseAttention stats - %s", holder.snapshot())
        guider.model_options = orig_model_options


class SparseAttentionModelNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SparseAttentionModel",
            display_name="Sparse Attention Model",
            category="advanced/model",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("method", options=SPARSE_ATTENTION_METHODS, default="pisa"),
                io.Combo.Input("apply_to", options=["all", "double", "single"], default="single", advanced=True),
                io.Int.Input("min_tokens", default=128, min=1, max=1000000, step=1, advanced=True),
                io.Int.Input("max_tokens", default=1000000, min=1, max=1000000, step=1, advanced=True),
                io.Float.Input("density", default=0.15, min=0.01, max=1.0, step=0.01, advanced=True),
                io.Int.Input("block_size", default=64, min=1, max=1024, step=1, advanced=True),
                io.Float.Input("topk", default=0.25, min=0.01, max=1.0, step=0.01, advanced=True),
                io.Boolean.Input("verbose", default=False, advanced=True),
                io.Int.Input("precompile_tokens", default=0, min=0, max=10000000, step=1, advanced=True),
                io.Int.Input("precompile_heads", default=24, min=1, max=1024, step=1, advanced=True),
                io.Int.Input("precompile_head_dim", default=128, min=1, max=1024, step=1, advanced=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        method,
        apply_to,
        min_tokens,
        max_tokens,
        density,
        block_size,
        topk,
        verbose,
        precompile_tokens=0,
        precompile_heads=24,
        precompile_head_dim=128,
    ) -> io.NodeOutput:
        patched = model.clone()
        config = SparseAttentionConfig(
            method=method,
            apply_to=apply_to,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            density=density,
            block_size=block_size,
            topk=topk,
            verbose=verbose,
            precompile_tokens=precompile_tokens,
            precompile_heads=precompile_heads,
            precompile_head_dim=precompile_head_dim,
        )
        holder = SparseAttentionHolder(config)
        if holder.precompile_pisa():
            logging.info(
                "SparseAttention PISA precompiled - tokens: %d, heads: %d, head_dim: %d, block_size: %d",
                config.precompile_tokens,
                config.precompile_heads,
                config.precompile_head_dim,
                config.block_size,
            )
        patched.model_options.setdefault("transformer_options", {})
        patched.model_options["transformer_options"]["sparse_attention"] = holder
        patched.add_wrapper_with_key("outer_sample", f"sparse_attention_{method}", sparse_attention_sample_wrapper)
        return io.NodeOutput(patched)


class SparseAttentionExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SparseAttentionModelNode]


def comfy_entrypoint():
    return SparseAttentionExtension()
