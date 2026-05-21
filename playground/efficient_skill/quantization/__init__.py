"""Independent quantization skill helpers."""

from efficient_skill.quantization.selective_linear import (
    insert_selective_torchao_linear_quant,
    selective_torchao_linear_quant_node,
)
from efficient_skill.quantization.torchao_fp8_mxfp8 import (
    DEFAULT_FLUX_SKIP_MODULES,
    insert_torchao_fp8_dynamic,
    insert_torchao_mxfp8_dynamic,
    torchao_quantize_model_node,
)
from efficient_skill.quantization.torchao_nvfp4 import insert_torchao_nvfp4_dynamic


MASKED_QUANTIZATION_SKILLS = (
    "modelopt_tensorrt_fp8_fp4",
    "standalone_svdquant_linear",
    "nunchaku_extracted_linear",
    "nunchaku_svdquant_backend_spec",
)


__all__ = [
    "DEFAULT_FLUX_SKIP_MODULES",
    "MASKED_QUANTIZATION_SKILLS",
    "insert_torchao_fp8_dynamic",
    "insert_torchao_mxfp8_dynamic",
    "insert_torchao_nvfp4_dynamic",
    "insert_selective_torchao_linear_quant",
    "selective_torchao_linear_quant_node",
    "torchao_quantize_model_node",
]
