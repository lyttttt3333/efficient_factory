# TorchAO FP8 Dynamic

Status: active Flux skill.

Group: `quantization`

Implementation:

- Skill helper: `efficient_skill.quantization.insert_torchao_fp8_dynamic`
- Runtime node: `TorchAOQuantizeModel`
- TorchAO recipe: `float8_dynamic`
- Flux canvas: `draft_canvas.flux_schnell_quantization_canvas.build_flux_schnell_torchao_fp8_demo`
- Benchmark id: `torchao_fp8_dynamic`

What It Changes

This quantizes eligible DiT `torch.nn.Linear` modules with TorchAO dynamic FP8
activation plus FP8 weight quantization. It targets only the diffusion
transformer. Text encoder and VAE are intentionally outside this skill.

How To Embed

Insert after the split diffusion model loader and before `BasicGuider`.

```python
from efficient_skill.quantization import insert_torchao_fp8_dynamic
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_torchao_fp8_dynamic(
    workflow,
    model_ref=output_ref("1", 0),
    skip_modules="img_in,txt_in,time_in,vector_in,guidance_in,final_layer",
)
```

The helper adds a `TorchAOQuantizeModel` node and retargets downstream model
references to the quantized model output.

Direct Model Integration

For raw Python inference, apply `torchao.quantization.quantize_` to the DiT
module, not to text encoders or VAE. Use a filter that skips fragile input and
final projection modules. Keep quantization/build time outside DiT denoise wall
time when benchmarking.

When To Use

Use as the first FP8 smoke test. It is a generic PyTorch/TorchAO runtime path,
not diffusion-specific calibration.
