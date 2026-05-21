# LazyCache

Status: active Flux skill.

Group: `cache`

Implementation:

- Skill helper: `efficient_skill.cache.lazycache.insert_lazycache`
- Runtime node: `LazyCache`
- Flux canvas: `draft_canvas.flux_schnell_cache_canvas.build_flux_schnell_lazycache_demo`
- Benchmark id: `lazycache` in `benchmark.flux_schnell_benchmark.CACHE_BUILDERS`

What It Changes

LazyCache patches only the diffusion model denoise path. It is a simpler cache
variant that reuses an approximated cached output/change when the latent update
is small enough. It does not modify CLIP/T5 text encoding or VAE decode.

How To Embed

Insert LazyCache after the split DiT loader and before the model is passed into
the guider/sampler.

```python
from efficient_skill.cache.lazycache import insert_lazycache
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_lazycache(
    workflow,
    model_ref=output_ref("1", 0),
    reuse_threshold=0.2,
    start_percent=0.15,
    end_percent=0.95,
)
```

The helper retargets downstream model inputs to the patched model reference.

Direct Model Integration

For direct Python inference, apply the same idea to the DiT ModelPatcher: clone
the model patcher, attach the LazyCache holder in transformer options, and wrap
the predict-noise path inside the denoise loop. Reset cache state at sample
boundaries.

When To Use

Use this as a lightweight reuse baseline. It can be faster to test than more
structured cache schemes, but full DiT wall-time and image metrics decide
whether it is acceptable.
