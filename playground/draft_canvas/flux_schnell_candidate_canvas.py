from __future__ import annotations

from draft_canvas.flux_schnell_cache_canvas import build_flux_schnell_baseline
from efficient_skill.common.workflow import Workflow, clone_workflow
from model.flux_schnell import FLUX_SCHNELL_SPLIT


CANDIDATE_NAME = "executor_candidate"
CANDIDATE_METADATA = {
    "candidate_name": CANDIDATE_NAME,
    "stack": [],
    "notes": "Initial candidate is the unmodified Flux Schnell baseline. Executor owns this file and should wrap the model here.",
}


def prepare_candidate(
    *,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    max_sequence_length: int,
    sampler_name: str,
    scheduler: str,
) -> dict:
    return {
        "prepared": False,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "max_sequence_length": max_sequence_length,
        "sampler_name": sampler_name,
        "scheduler": scheduler,
    }


def clear_candidate_state() -> None:
    return None


def build_flux_schnell_candidate(
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
    filename_prefix: str = "flux_executor_candidate",
) -> Workflow:
    return clone_workflow(
        build_flux_schnell_baseline(
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
    )
