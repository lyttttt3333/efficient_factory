# TorchAO MXFP8 Dynamic

Status: active Flux skill.

Group: `quantization`

Implementation:

- Skill helper: `efficient_skill.quantization.insert_torchao_mxfp8_dynamic`
- Runtime node: `TorchAOQuantizeModel`
- TorchAO recipe: `mxfp8_dynamic`
- Flux canvas: `draft_canvas.flux_schnell_quantization_canvas.build_flux_schnell_torchao_mxfp8_demo`
- Benchmark id: `torchao_mxfp8_dynamic`

What It Changes

This applies TorchAO microscaling FP8 to eligible DiT Linear modules. It is a
Blackwell-oriented W8A8 path and keeps the rest of the Flux pipeline unchanged.

How To Embed

Insert immediately after loading the split diffusion model.

```python
from efficient_skill.quantization import insert_torchao_mxfp8_dynamic
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_torchao_mxfp8_dynamic(
    workflow,
    model_ref=output_ref("1", 0),
    skip_modules="img_in,txt_in,time_in,vector_in,guidance_in,final_layer",
)
```

Direct Model Integration

Patch the DiT with TorchAO MXFP8 dynamic activation/weight config. Do not
quantize T5/CLIP or VAE unless a separate experiment explicitly targets them.
Keep the initial quantization conversion out of denoise timing.

When To Use

Use on RTX 5090/Blackwell when testing MXFP8 quality and speed. If full DiT
wall time is slower than BF16, prefer the selective MXFP8 skill.
