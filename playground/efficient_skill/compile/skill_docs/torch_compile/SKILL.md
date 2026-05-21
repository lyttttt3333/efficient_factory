# Torch Compile Model

Status: optional helper, not part of the default 15 Flux benchmark skills.

Group: `compile`

Implementation:

- Skill helper: `efficient_skill.compile.torch_compile.insert_torch_compile`
- Runtime node: `TorchCompileModel`
- Default backend: `inductor`

What It Changes

This inserts a compile model patch around the split diffusion model. It is a
generic PyTorch compile helper, not a cache, quantization, or sparse-attention
method. It should be benchmarked separately because first-run compile cost can
dominate short Flux Schnell runs.

How To Embed

Insert after the split DiT loader and before the guider consumes the model.

```python
from efficient_skill.compile.torch_compile import insert_torch_compile
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_torch_compile(
    workflow,
    model_ref=output_ref("1", 0),
    backend="inductor",
)
```

Direct Model Integration

For raw Python inference, compile the DiT forward path or the hot callable used
inside denoising, not text encoder or VAE, unless a separate experiment targets
those components. Warm up enough iterations to separate compile time from steady
state DiT denoise time.

When To Use

Use this as an optional composition helper after a single technique is already
working. Always report whether timing includes compile overhead.
