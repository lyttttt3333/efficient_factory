from __future__ import annotations


_MASKED_REASON = (
    "modelopt_tensorrt_fp8_fp4 is masked by request. "
    "It is an external TensorRT/ModelOpt engine path, not an active Flux quantization skill."
)


def modelopt_tensorrt_backend_spec(
    precision: str = "fp8",
    engine_dir: str = "benchmark/artifacts/tensorrt_engines",
    calibration_prompts: int = 16,
    quantize_attention: bool = True,
    cache_diffusion: bool = False,
) -> dict:
    """Return an external backend spec for TensorRT ModelOpt quantization.

    TensorRT/ModelOpt is an export-and-build route: it inserts QDQ nodes, builds
    a TensorRT engine, then runs that engine. That build step must stay separate
    from DiT denoise wall-time measurement.
    """
    raise RuntimeError(_MASKED_REASON)
    if precision not in {"fp8", "fp4"}:
        raise ValueError(f"Unsupported TensorRT precision: {precision}")
    return {
        "backend": "modelopt_tensorrt",
        "precision": precision,
        "engine_dir": engine_dir,
        "calibration_prompts": calibration_prompts,
        "quantize_attention": quantize_attention,
        "cache_diffusion": cache_diffusion,
        "requires": ["modelopt", "tensorrt"],
        "model_scope": "diffusion_transformer",
    }
