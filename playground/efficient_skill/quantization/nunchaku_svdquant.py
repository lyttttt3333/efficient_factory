from __future__ import annotations


_MASKED_REASON = (
    "nunchaku_svdquant_backend_spec is masked by request. "
    "It is an external checkpoint/backend path, not an active Flux quantization skill."
)


def nunchaku_svdquant_backend_spec(
    checkpoint: str,
    precision: str = "nvfp4",
    rank: int = 32,
    cpu_offload: bool = False,
) -> dict:
    """Return an external backend spec for Nunchaku/SVDQuant Flux inference.

    Nunchaku is not a direct in-place ModelPatcher transform in this repo. It
    loads an SVDQuant checkpoint and runs a separate fused W4A4 plus low-rank
    branch engine. Keep it as a backend spec so benchmark agents can route to a
    dedicated Nunchaku runner without confusing it with local Comfy model patches.
    """
    raise RuntimeError(_MASKED_REASON)
    if precision not in {"int4", "nvfp4", "fp4"}:
        raise ValueError(f"Unsupported SVDQuant precision: {precision}")
    return {
        "backend": "nunchaku_svdquant",
        "checkpoint": checkpoint,
        "precision": precision,
        "rank": rank,
        "cpu_offload": cpu_offload,
        "requires": ["nunchaku", "SVDQuant checkpoint"],
        "model_scope": "diffusion_transformer",
    }
