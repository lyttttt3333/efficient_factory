from __future__ import annotations

from efficient_skill.common.workflow import Workflow, next_node_id, output_ref, replace_model_input


DEFAULT_FLUX_SKIP_MODULES = "img_in,txt_in,time_in,vector_in,guidance_in,final_layer"


def torchao_quantize_model_node(
    model_ref: list,
    recipe: str,
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
    cache_quantized_model: bool = True,
) -> dict:
    return {
        "class_type": "TorchAOQuantizeModel",
        "inputs": {
            "model": model_ref,
            "recipe": recipe,
            "skip_modules": skip_modules,
            "cache_quantized_model": cache_quantized_model,
        },
    }


def insert_torchao_fp8_dynamic(
    workflow: Workflow,
    model_ref: list,
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
    cache_quantized_model: bool = True,
) -> tuple[Workflow, list]:
    node_id = next_node_id(workflow)
    workflow[node_id] = torchao_quantize_model_node(
        model_ref=model_ref,
        recipe="float8_dynamic",
        skip_modules=skip_modules,
        cache_quantized_model=cache_quantized_model,
    )
    new_model_ref = output_ref(node_id, 0)
    replace_model_input(workflow, old_ref=model_ref, new_ref=new_model_ref)
    workflow[node_id]["inputs"]["model"] = model_ref
    return workflow, new_model_ref


def insert_torchao_mxfp8_dynamic(
    workflow: Workflow,
    model_ref: list,
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
    cache_quantized_model: bool = True,
) -> tuple[Workflow, list]:
    node_id = next_node_id(workflow)
    workflow[node_id] = torchao_quantize_model_node(
        model_ref=model_ref,
        recipe="mxfp8_dynamic",
        skip_modules=skip_modules,
        cache_quantized_model=cache_quantized_model,
    )
    new_model_ref = output_ref(node_id, 0)
    replace_model_input(workflow, old_ref=model_ref, new_ref=new_model_ref)
    workflow[node_id]["inputs"]["model"] = model_ref
    return workflow, new_model_ref
