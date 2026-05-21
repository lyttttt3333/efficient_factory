# TeaCache

Status: active Flux skill.

Group: `cache`

Implementation:

- Skill helper: `efficient_skill.cache.teacache.insert_teacache`
- Runtime node: `TeaCache`
- Flux canvas: `draft_canvas.flux_schnell_cache_canvas.build_flux_schnell_teacache_demo`
- Benchmark id: `teacache` in `benchmark.flux_schnell_benchmark.CACHE_BUILDERS`

What It Changes

TeaCache patches the Flux DiT denoise loop. It tracks the relative L1 change of
the modulated DiT input, accumulates the Flux polynomial estimate, and reuses a
previous residual while the accumulated distance stays below `rel_l1_thresh`.

How To Embed

Insert TeaCache directly after loading the split diffusion model.

```python
from efficient_skill.cache.teacache import insert_teacache
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_teacache(
    workflow,
    model_ref=output_ref("1", 0),
    rel_l1_thresh=20.0,
    start_percent=0.0,
    end_percent=1.0,
    max_skip_steps=2,
    cache_device="default",
)
```

The canvas default is intentionally aggressive for the small Flux Schnell demo.
Retune `rel_l1_thresh` before treating it as a general quality setting.

Direct Model Integration

Attach a TeaCache holder to the DiT model options and wrap the outer denoise
sample call. The holder must see every denoise timestep in order, and its
residual state must be reset after each generated sample.

When To Use

Use TeaCache when you want step skipping tied to Flux-specific hidden-state
change rather than a simple fixed interval. Always report skipped steps,
DiT-only wall time, and image distance against the same seed baseline.
