from __future__ import annotations

from efficient_skill.common.workflow import Workflow, next_node_id, output_ref, replace_model_input
from efficient_skill.quantization.torchao_fp8_mxfp8 import DEFAULT_FLUX_SKIP_MODULES


def selective_torchao_linear_quant_node(
    model_ref: list,
    recipe: str = "nvfp4_dynamic",
    min_speedup: float = 1.05,
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
    warmup_runs: int = 3,
    benchmark_runs: int = 8,
    benchmark_loops: int = 4,
    cache_quantized_model: bool = True,
    verbose: bool = False,
) -> dict:
    return {
        "class_type": "SelectiveTorchAOQuantizeModel",
        "inputs": {
            "model": model_ref,
            "recipe": recipe,
            "min_speedup": min_speedup,
            "skip_modules": skip_modules,
            "warmup_runs": warmup_runs,
            "benchmark_runs": benchmark_runs,
            "benchmark_loops": benchmark_loops,
            "cache_quantized_model": cache_quantized_model,
            "verbose": verbose,
        },
    }


def insert_selective_torchao_linear_quant(
    workflow: Workflow,
    model_ref: list,
    recipe: str = "nvfp4_dynamic",
    min_speedup: float = 1.05,
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
    warmup_runs: int = 3,
    benchmark_runs: int = 8,
    benchmark_loops: int = 4,
    cache_quantized_model: bool = True,
    verbose: bool = False,
) -> tuple[Workflow, list]:
    node_id = next_node_id(workflow)
    workflow[node_id] = selective_torchao_linear_quant_node(
        model_ref=model_ref,
        recipe=recipe,
        min_speedup=min_speedup,
        skip_modules=skip_modules,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        benchmark_loops=benchmark_loops,
        cache_quantized_model=cache_quantized_model,
        verbose=verbose,
    )
    new_model_ref = output_ref(node_id, 0)
    replace_model_input(workflow, old_ref=model_ref, new_ref=new_model_ref)
    workflow[node_id]["inputs"]["model"] = model_ref
    return workflow, new_model_ref
