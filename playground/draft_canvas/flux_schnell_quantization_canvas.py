from __future__ import annotations

from efficient_skill.common.workflow import Workflow, clone_workflow, output_ref
from efficient_skill.quantization import (
    DEFAULT_FLUX_SKIP_MODULES,
    MASKED_QUANTIZATION_SKILLS,
    insert_selective_torchao_linear_quant,
    insert_torchao_fp8_dynamic,
    insert_torchao_mxfp8_dynamic,
    insert_torchao_nvfp4_dynamic,
)
from draft_canvas.flux_schnell_cache_canvas import build_flux_schnell_baseline
from model.flux_schnell import FLUX_SCHNELL_SPLIT


def _build_quant_base(
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


def build_flux_schnell_torchao_fp8_demo(
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
    filename_prefix: str = "flux_torchao_fp8",
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
) -> Workflow:
    workflow = _build_quant_base(
        prompt, seed, diffusion_name, t5xxl_name, clip_l_name, vae_name,
        width, height, steps, guidance, sampler_name, scheduler, filename_prefix,
    )
    insert_torchao_fp8_dynamic(workflow, model_ref=output_ref("1", 0), skip_modules=skip_modules)
    return workflow


def build_flux_schnell_torchao_mxfp8_demo(
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
    filename_prefix: str = "flux_torchao_mxfp8",
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
) -> Workflow:
    workflow = _build_quant_base(
        prompt, seed, diffusion_name, t5xxl_name, clip_l_name, vae_name,
        width, height, steps, guidance, sampler_name, scheduler, filename_prefix,
    )
    insert_torchao_mxfp8_dynamic(workflow, model_ref=output_ref("1", 0), skip_modules=skip_modules)
    return workflow


def build_flux_schnell_torchao_nvfp4_demo(
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
    filename_prefix: str = "flux_torchao_nvfp4",
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
) -> Workflow:
    workflow = _build_quant_base(
        prompt, seed, diffusion_name, t5xxl_name, clip_l_name, vae_name,
        width, height, steps, guidance, sampler_name, scheduler, filename_prefix,
    )
    insert_torchao_nvfp4_dynamic(workflow, model_ref=output_ref("1", 0), skip_modules=skip_modules)
    return workflow


def build_flux_schnell_selective_torchao_quant_demo(
    prompt: str,
    seed: int,
    recipe: str = "nvfp4_dynamic",
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
    filename_prefix: str = "flux_selective_torchao_quant",
    min_speedup: float = 1.05,
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
    warmup_runs: int = 3,
    benchmark_runs: int = 8,
    benchmark_loops: int = 4,
) -> Workflow:
    workflow = _build_quant_base(
        prompt, seed, diffusion_name, t5xxl_name, clip_l_name, vae_name,
        width, height, steps, guidance, sampler_name, scheduler, filename_prefix,
    )
    insert_selective_torchao_linear_quant(
        workflow,
        model_ref=output_ref("1", 0),
        recipe=recipe,
        min_speedup=min_speedup,
        skip_modules=skip_modules,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        benchmark_loops=benchmark_loops,
        verbose=False,
    )
    return workflow


def build_flux_schnell_selective_torchao_fp8_demo(**kwargs) -> Workflow:
    kwargs.setdefault("filename_prefix", "flux_selective_torchao_fp8")
    return build_flux_schnell_selective_torchao_quant_demo(recipe="float8_dynamic", **kwargs)


def build_flux_schnell_selective_torchao_mxfp8_demo(**kwargs) -> Workflow:
    kwargs.setdefault("filename_prefix", "flux_selective_torchao_mxfp8")
    return build_flux_schnell_selective_torchao_quant_demo(recipe="mxfp8_dynamic", **kwargs)


def build_flux_schnell_selective_torchao_nvfp4_demo(**kwargs) -> Workflow:
    kwargs.setdefault("filename_prefix", "flux_selective_torchao_nvfp4")
    return build_flux_schnell_selective_torchao_quant_demo(recipe="nvfp4_dynamic", **kwargs)


QUANTIZATION_WORKFLOW_BUILDERS = {
    "torchao_fp8_dynamic": build_flux_schnell_torchao_fp8_demo,
    "torchao_mxfp8_dynamic": build_flux_schnell_torchao_mxfp8_demo,
    "torchao_nvfp4_dynamic": build_flux_schnell_torchao_nvfp4_demo,
    "selective_torchao_fp8": build_flux_schnell_selective_torchao_fp8_demo,
    "selective_torchao_mxfp8": build_flux_schnell_selective_torchao_mxfp8_demo,
    "selective_torchao_nvfp4": build_flux_schnell_selective_torchao_nvfp4_demo,
}


QUANTIZATION_EXTERNAL_BACKENDS = {}
QUANTIZATION_MASKED_SKILLS = MASKED_QUANTIZATION_SKILLS
