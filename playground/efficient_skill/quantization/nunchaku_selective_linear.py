from __future__ import annotations

from efficient_skill.common.workflow import Workflow, next_node_id, output_ref, replace_model_input
from efficient_skill.quantization.torchao_fp8_mxfp8 import DEFAULT_FLUX_SKIP_MODULES


_MASKED_REASON = (
    "nunchaku_extracted_linear is masked by request. "
    "The extracted Nunchaku Linear path is retained in source but is not an active Flux quantization skill."
)


def nunchaku_extracted_linear_node(
    model_ref: list,
    precision: str = "int4",
    rank: int = 16,
    min_speedup: float = 1.05,
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
    max_modules: int = 16,
    warmup_runs: int = 2,
    benchmark_runs: int = 4,
    benchmark_loops: int = 2,
    svd_niter: int = 1,
    cache_quantized_model: bool = True,
    verbose: bool = False,
) -> dict:
    raise RuntimeError(_MASKED_REASON)
    return {
        "class_type": "NunchakuExtractedLinearModel",
        "inputs": {
            "model": model_ref,
            "precision": precision,
            "rank": rank,
            "min_speedup": min_speedup,
            "skip_modules": skip_modules,
            "max_modules": max_modules,
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
            "benchmark_loops": benchmark_loops,
            "svd_niter": svd_niter,
            "cache_quantized_model": cache_quantized_model,
            "verbose": verbose,
        },
    }


def insert_nunchaku_extracted_linear(
    workflow: Workflow,
    model_ref: list,
    precision: str = "int4",
    rank: int = 16,
    min_speedup: float = 1.05,
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
    max_modules: int = 16,
    warmup_runs: int = 2,
    benchmark_runs: int = 4,
    benchmark_loops: int = 2,
    svd_niter: int = 1,
    cache_quantized_model: bool = True,
    verbose: bool = False,
) -> tuple[Workflow, list]:
    raise RuntimeError(_MASKED_REASON)
    node_id = next_node_id(workflow)
    workflow[node_id] = nunchaku_extracted_linear_node(
        model_ref=model_ref,
        precision=precision,
        rank=rank,
        min_speedup=min_speedup,
        skip_modules=skip_modules,
        max_modules=max_modules,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        benchmark_loops=benchmark_loops,
        svd_niter=svd_niter,
        cache_quantized_model=cache_quantized_model,
        verbose=verbose,
    )
    new_model_ref = output_ref(node_id, 0)
    replace_model_input(workflow, old_ref=model_ref, new_ref=new_model_ref)
    workflow[node_id]["inputs"]["model"] = model_ref
    return workflow, new_model_ref
