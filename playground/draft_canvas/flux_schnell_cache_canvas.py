from __future__ import annotations

from efficient_skill.cache.easycache import insert_easycache
from efficient_skill.cache.lazycache import insert_lazycache
from efficient_skill.cache.teacache import insert_teacache
from efficient_skill.cache.periodic_reuse import insert_periodic_reuse_cache
from efficient_skill.cache.similarity_reuse import insert_similarity_reuse_cache
from efficient_skill.cache.delta_reuse import insert_delta_reuse_cache
from efficient_skill.cache.ema_reuse import insert_ema_reuse_cache
from efficient_skill.common.workflow import Workflow, clone_workflow, output_ref
from draft_canvas.split_loaders import (
    diffusion_model_loader_node,
    dual_clip_loader_node,
    vae_loader_node,
)
from model.flux_schnell import FLUX_SCHNELL_SPLIT


def build_flux_schnell_baseline(
    prompt: str,
    seed: int,
    diffusion_name: str = FLUX_SCHNELL_SPLIT.diffusion.filename,
    t5xxl_name: str = FLUX_SCHNELL_SPLIT.t5xxl.filename,
    clip_l_name: str = FLUX_SCHNELL_SPLIT.clip_l.filename,
    vae_name: str = FLUX_SCHNELL_SPLIT.vae.filename,
    diffusion_weight_dtype: str = "default",
    clip_device: str = "default",
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    guidance: float = 3.5,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    filename_prefix: str = "flux_baseline",
) -> Workflow:
    return {
        "1": diffusion_model_loader_node(diffusion_name, weight_dtype=diffusion_weight_dtype),
        "2": dual_clip_loader_node(t5xxl_name, clip_l_name, clip_type="flux", device=clip_device),
        "3": vae_loader_node(vae_name),
        "4": {
            "class_type": "CLIPTextEncodeFlux",
            "inputs": {
                "clip": output_ref("2", 0),
                "clip_l": prompt,
                "t5xxl": prompt,
                "guidance": guidance,
            },
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
        },
        "6": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": output_ref("1", 0),
                "conditioning": output_ref("4", 0),
            },
        },
        "7": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": sampler_name,
            },
        },
        "8": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": output_ref("1", 0),
                "scheduler": scheduler,
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "9": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": seed,
            },
        },
        "10": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": output_ref("9", 0),
                "guider": output_ref("6", 0),
                "sampler": output_ref("7", 0),
                "sigmas": output_ref("8", 0),
                "latent_image": output_ref("5", 0),
            },
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": output_ref("10", 0),
                "vae": output_ref("3", 0),
            },
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {
                "images": output_ref("11", 0),
                "filename_prefix": filename_prefix,
            },
        },
    }


def build_flux_schnell_easycache_demo(
    prompt: str,
    seed: int,
    diffusion_name: str = FLUX_SCHNELL_SPLIT.diffusion.filename,
    t5xxl_name: str = FLUX_SCHNELL_SPLIT.t5xxl.filename,
    clip_l_name: str = FLUX_SCHNELL_SPLIT.clip_l.filename,
    vae_name: str = FLUX_SCHNELL_SPLIT.vae.filename,
    diffusion_weight_dtype: str = "default",
    clip_device: str = "default",
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    guidance: float = 3.5,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    filename_prefix: str = "flux_easycache",
    reuse_threshold: float = 0.2,
    start_percent: float = 0.15,
    end_percent: float = 0.95,
) -> Workflow:
    workflow = build_flux_schnell_baseline(
        prompt=prompt,
        seed=seed,
        diffusion_name=diffusion_name,
        t5xxl_name=t5xxl_name,
        clip_l_name=clip_l_name,
        vae_name=vae_name,
        diffusion_weight_dtype=diffusion_weight_dtype,
        clip_device=clip_device,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        sampler_name=sampler_name,
        scheduler=scheduler,
        filename_prefix=filename_prefix,
    )
    workflow = clone_workflow(workflow)
    insert_easycache(
        workflow,
        model_ref=output_ref("1", 0),
        reuse_threshold=reuse_threshold,
        start_percent=start_percent,
        end_percent=end_percent,
        verbose=False,
    )
    return workflow


def build_flux_schnell_lazycache_demo(
    prompt: str,
    seed: int,
    diffusion_name: str = FLUX_SCHNELL_SPLIT.diffusion.filename,
    t5xxl_name: str = FLUX_SCHNELL_SPLIT.t5xxl.filename,
    clip_l_name: str = FLUX_SCHNELL_SPLIT.clip_l.filename,
    vae_name: str = FLUX_SCHNELL_SPLIT.vae.filename,
    diffusion_weight_dtype: str = "default",
    clip_device: str = "default",
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    guidance: float = 3.5,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    filename_prefix: str = "flux_lazycache",
    reuse_threshold: float = 0.2,
    start_percent: float = 0.15,
    end_percent: float = 0.95,
) -> Workflow:
    workflow = build_flux_schnell_baseline(
        prompt=prompt,
        seed=seed,
        diffusion_name=diffusion_name,
        t5xxl_name=t5xxl_name,
        clip_l_name=clip_l_name,
        vae_name=vae_name,
        diffusion_weight_dtype=diffusion_weight_dtype,
        clip_device=clip_device,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        sampler_name=sampler_name,
        scheduler=scheduler,
        filename_prefix=filename_prefix,
    )
    workflow = clone_workflow(workflow)
    insert_lazycache(
        workflow,
        model_ref=output_ref("1", 0),
        reuse_threshold=reuse_threshold,
        start_percent=start_percent,
        end_percent=end_percent,
        verbose=False,
    )
    return workflow


def build_flux_schnell_teacache_demo(
    prompt: str,
    seed: int,
    diffusion_name: str = FLUX_SCHNELL_SPLIT.diffusion.filename,
    t5xxl_name: str = FLUX_SCHNELL_SPLIT.t5xxl.filename,
    clip_l_name: str = FLUX_SCHNELL_SPLIT.clip_l.filename,
    vae_name: str = FLUX_SCHNELL_SPLIT.vae.filename,
    diffusion_weight_dtype: str = "default",
    clip_device: str = "default",
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    guidance: float = 3.5,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    filename_prefix: str = "flux_teacache",
    rel_l1_thresh: float = 20.0,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    max_skip_steps: int = 2,
) -> Workflow:
    workflow = build_flux_schnell_baseline(
        prompt=prompt,
        seed=seed,
        diffusion_name=diffusion_name,
        t5xxl_name=t5xxl_name,
        clip_l_name=clip_l_name,
        vae_name=vae_name,
        diffusion_weight_dtype=diffusion_weight_dtype,
        clip_device=clip_device,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        sampler_name=sampler_name,
        scheduler=scheduler,
        filename_prefix=filename_prefix,
    )
    workflow = clone_workflow(workflow)
    insert_teacache(
        workflow,
        model_ref=output_ref("1", 0),
        rel_l1_thresh=rel_l1_thresh,
        start_percent=start_percent,
        end_percent=end_percent,
        max_skip_steps=max_skip_steps,
        cache_device="default",
        verbose=False,
    )
    return workflow


def build_flux_schnell_periodic_reuse_demo(
    prompt: str,
    seed: int,
    diffusion_name: str = FLUX_SCHNELL_SPLIT.diffusion.filename,
    t5xxl_name: str = FLUX_SCHNELL_SPLIT.t5xxl.filename,
    clip_l_name: str = FLUX_SCHNELL_SPLIT.clip_l.filename,
    vae_name: str = FLUX_SCHNELL_SPLIT.vae.filename,
    diffusion_weight_dtype: str = "default",
    clip_device: str = "default",
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    guidance: float = 3.5,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    filename_prefix: str = "flux_periodic_reuse",
    interval: int = 2,
    warmup_steps: int = 1,
    max_skip_steps: int = 1,
) -> Workflow:
    workflow = clone_workflow(build_flux_schnell_baseline(
        prompt=prompt,
        seed=seed,
        diffusion_name=diffusion_name,
        t5xxl_name=t5xxl_name,
        clip_l_name=clip_l_name,
        vae_name=vae_name,
        diffusion_weight_dtype=diffusion_weight_dtype,
        clip_device=clip_device,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        sampler_name=sampler_name,
        scheduler=scheduler,
        filename_prefix=filename_prefix,
    ))
    insert_periodic_reuse_cache(
        workflow,
        model_ref=output_ref("1", 0),
        interval=interval,
        warmup_steps=warmup_steps,
        max_skip_steps=max_skip_steps,
        verbose=False,
    )
    return workflow


def build_flux_schnell_similarity_reuse_demo(
    prompt: str,
    seed: int,
    diffusion_name: str = FLUX_SCHNELL_SPLIT.diffusion.filename,
    t5xxl_name: str = FLUX_SCHNELL_SPLIT.t5xxl.filename,
    clip_l_name: str = FLUX_SCHNELL_SPLIT.clip_l.filename,
    vae_name: str = FLUX_SCHNELL_SPLIT.vae.filename,
    diffusion_weight_dtype: str = "default",
    clip_device: str = "default",
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    guidance: float = 3.5,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    filename_prefix: str = "flux_similarity_reuse",
    similarity_threshold: float = 1.00,
    subsample_factor: int = 8,
    warmup_steps: int = 1,
    max_skip_steps: int = 1,
) -> Workflow:
    workflow = clone_workflow(build_flux_schnell_baseline(
        prompt=prompt,
        seed=seed,
        diffusion_name=diffusion_name,
        t5xxl_name=t5xxl_name,
        clip_l_name=clip_l_name,
        vae_name=vae_name,
        diffusion_weight_dtype=diffusion_weight_dtype,
        clip_device=clip_device,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        sampler_name=sampler_name,
        scheduler=scheduler,
        filename_prefix=filename_prefix,
    ))
    insert_similarity_reuse_cache(
        workflow,
        model_ref=output_ref("1", 0),
        similarity_threshold=similarity_threshold,
        subsample_factor=subsample_factor,
        warmup_steps=warmup_steps,
        max_skip_steps=max_skip_steps,
        verbose=False,
    )
    return workflow


def build_flux_schnell_delta_reuse_demo(
    prompt: str,
    seed: int,
    diffusion_name: str = FLUX_SCHNELL_SPLIT.diffusion.filename,
    t5xxl_name: str = FLUX_SCHNELL_SPLIT.t5xxl.filename,
    clip_l_name: str = FLUX_SCHNELL_SPLIT.clip_l.filename,
    vae_name: str = FLUX_SCHNELL_SPLIT.vae.filename,
    diffusion_weight_dtype: str = "default",
    clip_device: str = "default",
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    guidance: float = 3.5,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    filename_prefix: str = "flux_delta_reuse",
    delta_scale: float = 1.0,
    interval: int = 2,
    warmup_steps: int = 2,
    max_skip_steps: int = 1,
) -> Workflow:
    workflow = clone_workflow(build_flux_schnell_baseline(
        prompt=prompt,
        seed=seed,
        diffusion_name=diffusion_name,
        t5xxl_name=t5xxl_name,
        clip_l_name=clip_l_name,
        vae_name=vae_name,
        diffusion_weight_dtype=diffusion_weight_dtype,
        clip_device=clip_device,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        sampler_name=sampler_name,
        scheduler=scheduler,
        filename_prefix=filename_prefix,
    ))
    insert_delta_reuse_cache(
        workflow,
        model_ref=output_ref("1", 0),
        delta_scale=delta_scale,
        interval=interval,
        warmup_steps=warmup_steps,
        max_skip_steps=max_skip_steps,
        verbose=False,
    )
    return workflow


def build_flux_schnell_ema_reuse_demo(
    prompt: str,
    seed: int,
    diffusion_name: str = FLUX_SCHNELL_SPLIT.diffusion.filename,
    t5xxl_name: str = FLUX_SCHNELL_SPLIT.t5xxl.filename,
    clip_l_name: str = FLUX_SCHNELL_SPLIT.clip_l.filename,
    vae_name: str = FLUX_SCHNELL_SPLIT.vae.filename,
    diffusion_weight_dtype: str = "default",
    clip_device: str = "default",
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    guidance: float = 3.5,
    sampler_name: str = "euler",
    scheduler: str = "simple",
    filename_prefix: str = "flux_ema_reuse",
    ema_decay: float = 0.5,
    interval: int = 2,
    warmup_steps: int = 1,
    max_skip_steps: int = 1,
) -> Workflow:
    workflow = clone_workflow(build_flux_schnell_baseline(
        prompt=prompt,
        seed=seed,
        diffusion_name=diffusion_name,
        t5xxl_name=t5xxl_name,
        clip_l_name=clip_l_name,
        vae_name=vae_name,
        diffusion_weight_dtype=diffusion_weight_dtype,
        clip_device=clip_device,
        width=width,
        height=height,
        steps=steps,
        guidance=guidance,
        sampler_name=sampler_name,
        scheduler=scheduler,
        filename_prefix=filename_prefix,
    ))
    insert_ema_reuse_cache(
        workflow,
        model_ref=output_ref("1", 0),
        ema_decay=ema_decay,
        interval=interval,
        warmup_steps=warmup_steps,
        max_skip_steps=max_skip_steps,
        verbose=False,
    )
    return workflow


CACHE_WORKFLOW_BUILDERS = {
    "easycache": build_flux_schnell_easycache_demo,
    "lazycache": build_flux_schnell_lazycache_demo,
    "teacache": build_flux_schnell_teacache_demo,
    "periodic_reuse": build_flux_schnell_periodic_reuse_demo,
    "similarity_reuse": build_flux_schnell_similarity_reuse_demo,
    "delta_reuse": build_flux_schnell_delta_reuse_demo,
    "ema_reuse": build_flux_schnell_ema_reuse_demo,
}
