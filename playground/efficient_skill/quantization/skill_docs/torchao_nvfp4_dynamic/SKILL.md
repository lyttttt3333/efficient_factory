# TorchAO NVFP4 Dynamic

Status: active Flux skill.

Group: `quantization`

Implementation:

- Skill helper: `efficient_skill.quantization.insert_torchao_nvfp4_dynamic`
- Runtime node: `TorchAOQuantizeModel`
- TorchAO recipe: `nvfp4_dynamic`
- Flux canvas: `draft_canvas.flux_schnell_quantization_canvas.build_flux_schnell_torchao_nvfp4_demo`
- Benchmark id: `torchao_nvfp4_dynamic`

What It Changes

This applies TorchAO NVFP4 dynamic activation plus NVFP4 weight quantization to
eligible DiT Linear modules. It is the most aggressive active quantization path
and has the highest quality risk.

How To Embed

Insert after the split DiT loader.

```python
from efficient_skill.quantization import insert_torchao_nvfp4_dynamic
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_torchao_nvfp4_dynamic(
    workflow,
    model_ref=output_ref("1", 0),
    skip_modules="img_in,txt_in,time_in,vector_in,guidance_in,final_layer",
)
```

Direct Model Integration

Use TorchAO's NVFP4 dynamic config on DiT Linear layers only. The active runtime
expects CUDA/Blackwell support. Keep unsupported modules in high precision via a
module-name filter.

When To Use

Use when speed is prioritized and image loss can be measured against the fixed
baseline. Compare against `selective_torchao_nvfp4` before accepting a full
model-wide NVFP4 patch.
