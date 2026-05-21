# Selective TorchAO FP8

Status: active Flux skill.

Group: `quantization`

Implementation:

- Skill helper: `efficient_skill.quantization.insert_selective_torchao_linear_quant`
- Runtime node: `SelectiveTorchAOQuantizeModel`
- TorchAO recipe: `float8_dynamic`
- Flux canvas: `draft_canvas.flux_schnell_quantization_canvas.build_flux_schnell_selective_torchao_fp8_demo`
- Benchmark id: `selective_torchao_fp8`

What It Changes

This wraps eligible DiT Linear modules, observes their actual runtime input
shape, benchmarks FP8 vs high precision for that shape, and quantizes only
modules whose isolated Linear speedup is at least `min_speedup`.

How To Embed

Insert the selective quantizer after the split DiT loader.

```python
from efficient_skill.quantization import insert_selective_torchao_linear_quant
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_selective_torchao_linear_quant(
    workflow,
    model_ref=output_ref("1", 0),
    recipe="float8_dynamic",
    min_speedup=1.05,
    warmup_runs=3,
    benchmark_runs=8,
    benchmark_loops=4,
)
```

Direct Model Integration

Wrap DiT Linear layers with a small decision module. On first forward for each
shape, benchmark the original Linear and a TorchAO-quantized copy with warmup,
then quantize the real module only if the measured isolated speedup passes the
threshold. Store decisions by shape.

When To Use

Use when model-wide FP8 is inconsistent. The per-shape decision is only a local
kernel proxy; final acceptance still comes from full DiT wall time and quality.
