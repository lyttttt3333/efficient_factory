from __future__ import annotations

from efficient_skill.common.workflow import Workflow, next_node_id, output_ref, replace_model_input
from efficient_skill.quantization.torchao_fp8_mxfp8 import DEFAULT_FLUX_SKIP_MODULES, torchao_quantize_model_node


def insert_torchao_nvfp4_dynamic(
    workflow: Workflow,
    model_ref: list,
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
    cache_quantized_model: bool = True,
) -> tuple[Workflow, list]:
    node_id = next_node_id(workflow)
    workflow[node_id] = torchao_quantize_model_node(
        model_ref=model_ref,
        recipe="nvfp4_dynamic",
        skip_modules=skip_modules,
        cache_quantized_model=cache_quantized_model,
    )
    new_model_ref = output_ref(node_id, 0)
    replace_model_input(workflow, old_ref=model_ref, new_ref=new_model_ref)
    workflow[node_id]["inputs"]["model"] = model_ref
    return workflow, new_model_ref
