# Selective TorchAO MXFP8

Status: active Flux skill.

Group: `quantization`

Implementation:

- Skill helper: `efficient_skill.quantization.insert_selective_torchao_linear_quant`
- Runtime node: `SelectiveTorchAOQuantizeModel`
- TorchAO recipe: `mxfp8_dynamic`
- Flux canvas: `draft_canvas.flux_schnell_quantization_canvas.build_flux_schnell_selective_torchao_mxfp8_demo`
- Benchmark id: `selective_torchao_mxfp8`

What It Changes

This selectively applies TorchAO MXFP8 to DiT Linear modules. Each runtime
Linear shape is benchmarked with warmup, and only shapes that beat high
precision by `min_speedup` are quantized.

How To Embed

Patch the DiT model after the split loader.

```python
from efficient_skill.quantization import insert_selective_torchao_linear_quant
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_selective_torchao_linear_quant(
    workflow,
    model_ref=output_ref("1", 0),
    recipe="mxfp8_dynamic",
    min_speedup=1.05,
)
```

Direct Model Integration

Use a wrapper around DiT Linear modules that benchmarks BF16/high precision vs
TorchAO MXFP8 on first real shape. Quantize only the modules mapped to faster
shapes and keep all other modules unchanged.

When To Use

Use this as the safer Blackwell FP8 path when full MXFP8 conversion is slower or
when only a subset of Flux Linear shapes benefits.
