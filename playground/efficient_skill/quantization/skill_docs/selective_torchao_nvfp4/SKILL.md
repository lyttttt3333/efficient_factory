# Selective TorchAO NVFP4

Status: active Flux skill.

Group: `quantization`

Implementation:

- Skill helper: `efficient_skill.quantization.insert_selective_torchao_linear_quant`
- Runtime node: `SelectiveTorchAOQuantizeModel`
- TorchAO recipe: `nvfp4_dynamic`
- Flux canvas: `draft_canvas.flux_schnell_quantization_canvas.build_flux_schnell_selective_torchao_nvfp4_demo`
- Benchmark id: `selective_torchao_nvfp4`

What It Changes

This selectively applies TorchAO NVFP4 to DiT Linear modules. It benchmarks the
actual runtime shape before mutating the module, then keeps non-winning modules
in high precision.

How To Embed

Insert after the split DiT loader.

```python
from efficient_skill.quantization import insert_selective_torchao_linear_quant
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_selective_torchao_linear_quant(
    workflow,
    model_ref=output_ref("1", 0),
    recipe="nvfp4_dynamic",
    min_speedup=1.05,
)
```

Direct Model Integration

Wrap DiT Linear modules, benchmark high precision vs TorchAO NVFP4 at first
real input shape with CUDA warmup, and quantize only modules whose measured
speedup passes the threshold. Keep input projections and final layers skipped
unless a dedicated experiment proves otherwise.

When To Use

Use this when NVFP4 gives speed but full-model quality loss is too large. Report
selected module count, kept module count, full DiT wall-time speedup, and image
metrics.
