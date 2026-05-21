from __future__ import annotations

import math

from draft_canvas.flux_schnell_cache_canvas import build_flux_schnell_baseline
from efficient_skill.common.workflow import Workflow, clone_workflow, output_ref
from efficient_skill.sparse_attention import insert_pisa_sparse_attention, insert_spargeattn_sparse_attention
from model.flux_schnell import FLUX_SCHNELL_SPLIT


def _build_sparse_base(
    prompt: str,
    seed: int,
    diffusion_name: str,
    t5xxl_name: str,
    clip_l_name: str,
    vae_name: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    sampler_name: str,
    scheduler: str,
    filename_prefix: str,
) -> Workflow:
    return clone_workflow(build_flux_schnell_baseline(
        prompt=prompt,
        seed=seed,
        diffusion_name=diffusion_name,
        t5xxl_name=t5xxl_name,
        clip_l_name=clip_l_name,
        vae_name=vae_name,
        diffusion_weight_dtype="default",
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        sampler_name=sampler_name,
        scheduler=scheduler,
        filename_prefix=filename_prefix,
    ))


def _common_kwargs(
    apply_to: str,
    min_tokens: int,
    max_tokens: int,
    verbose: bool,
) -> dict:
    return {
        "apply_to": apply_to,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "verbose": verbose,
    }


def _flux_attention_tokens(width: int, height: int, text_tokens: int = 256) -> int:
    latent_w = max(1, math.ceil(width / 16))
    latent_h = max(1, math.ceil(height / 16))
    return latent_w * latent_h + text_tokens


def build_flux_schnell_pisa_sparse_attention_demo(
    prompt: str,
    seed: int,
    diffusion_name: str = FLUX_SCHNELL_SPLIT.diffusion.filename,
    t5xxl_name: str = FLUX_SCHNELL_SPLIT.t5xxl.filename,
    clip_l_name: str = FLUX_SCHNELL_SPLIT.clip_l.filename,
    vae_name: str = FLUX_SCHNELL_SPLIT.vae.filename,
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    guidance: float = 3.5,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    filename_prefix: str = "flux_pisa_sparse_attention",
    apply_to: str = "single",
    min_tokens: int = 128,
    max_tokens: int = 1_000_000,
    density: float = 0.15,
    block_size: int = 128,
    precompile_tokens: int | None = None,
    precompile_heads: int = 24,
    precompile_head_dim: int = 128,
    verbose: bool = False,
) -> Workflow:
    workflow = _build_sparse_base(prompt, seed, diffusion_name, t5xxl_name, clip_l_name, vae_name, width, height, steps, guidance, sampler_name, scheduler, filename_prefix)
    if precompile_tokens is None:
        precompile_tokens = _flux_attention_tokens(width, height)
    insert_pisa_sparse_attention(
        workflow,
        output_ref("1", 0),
        **_common_kwargs(apply_to, min_tokens, max_tokens, verbose),
        density=density,
        block_size=block_size,
        precompile_tokens=precompile_tokens,
        precompile_heads=precompile_heads,
        precompile_head_dim=precompile_head_dim,
    )
    return workflow


def build_flux_schnell_spargeattn_sparse_attention_demo(
    prompt: str,
    seed: int,
    diffusion_name: str = FLUX_SCHNELL_SPLIT.diffusion.filename,
    t5xxl_name: str = FLUX_SCHNELL_SPLIT.t5xxl.filename,
    clip_l_name: str = FLUX_SCHNELL_SPLIT.clip_l.filename,
    vae_name: str = FLUX_SCHNELL_SPLIT.vae.filename,
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    guidance: float = 3.5,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    filename_prefix: str = "flux_spargeattn_sparse_attention",
    apply_to: str = "single",
    min_tokens: int = 128,
    max_tokens: int = 1_000_000,
    topk: float = 0.25,
    verbose: bool = False,
) -> Workflow:
    workflow = _build_sparse_base(prompt, seed, diffusion_name, t5xxl_name, clip_l_name, vae_name, width, height, steps, guidance, sampler_name, scheduler, filename_prefix)
    insert_spargeattn_sparse_attention(
        workflow,
        output_ref("1", 0),
        **_common_kwargs(apply_to, min_tokens, max_tokens, verbose),
        topk=topk,
    )
    return workflow


SPARSE_ATTENTION_WORKFLOW_BUILDERS = {
    "pisa": build_flux_schnell_pisa_sparse_attention_demo,
    "spargeattn": build_flux_schnell_spargeattn_sparse_attention_demo,
}
