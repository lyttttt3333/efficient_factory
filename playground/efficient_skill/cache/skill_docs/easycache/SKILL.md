# EasyCache

Status: active Flux skill.

Group: `cache`

Implementation:

- Skill helper: `efficient_skill.cache.easycache.insert_easycache`
- Runtime node: `EasyCache`
- Flux canvas: `draft_canvas.flux_schnell_cache_canvas.build_flux_schnell_easycache_demo`
- Benchmark id: `easycache` in `benchmark.flux_schnell_benchmark.CACHE_BUILDERS`

What It Changes

EasyCache patches only the diffusion model path. It stores and reuses DiT
intermediate/output differences during denoising when the latent change is below
`reuse_threshold`. Text encoders, VAE decode, model download, and image saving
are not part of this skill.

How To Embed

In the workflow graph, insert EasyCache immediately after the split diffusion
model loader and before any consumer of that model, usually `BasicGuider`.

```python
from efficient_skill.cache.easycache import insert_easycache
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_easycache(
    workflow,
    model_ref=output_ref("1", 0),
    reuse_threshold=0.2,
    start_percent=0.15,
    end_percent=0.95,
)
```

`insert_easycache` adds an `EasyCache` model patch node, replaces downstream
references to the original model with the patched model, and keeps the cache
node input pointed at the original DiT loader output.

Direct Model Integration

For a raw Python DiT runner, mirror the runtime node behavior: clone the
ModelPatcher, attach an `EasyCacheHolder` to
`model_options["transformer_options"]["easycache"]`, then wrap the diffusion
forward/model sampling calls. The cache must live inside the denoise loop and
must be reset between samples.

When To Use

Use this as a broad cache baseline. Tune `reuse_threshold`,
`start_percent`, and `end_percent` against the fixed benchmark because speedup
comes from skipped DiT calls and quality loss depends strongly on the prompt and
noise seed.
