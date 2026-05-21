# Periodic Reuse Cache

Status: active Flux skill.

Group: `cache`

Implementation:

- Skill helper: `efficient_skill.cache.periodic_reuse.insert_periodic_reuse_cache`
- Runtime node: `PeriodicReuseCache`
- Flux canvas: `draft_canvas.flux_schnell_cache_canvas.build_flux_schnell_periodic_reuse_demo`
- Benchmark id: `periodic_reuse` in `benchmark.flux_schnell_benchmark.CACHE_BUILDERS`

What It Changes

Periodic reuse skips DiT calls on a fixed schedule after `warmup_steps`, then
returns the last cached denoise output. It is intentionally simple and useful as
a lower-bound control for cache/reuse experiments.

How To Embed

Insert the model patch after the split DiT loader.

```python
from efficient_skill.cache.periodic_reuse import insert_periodic_reuse_cache
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_periodic_reuse_cache(
    workflow,
    model_ref=output_ref("1", 0),
    interval=2,
    warmup_steps=1,
    max_skip_steps=1,
    start_percent=0.0,
    end_percent=1.0,
)
```

Direct Model Integration

Clone the DiT ModelPatcher, attach a reuse holder with mode `periodic`, and wrap
the predict-noise function. The holder decides at each timestep whether to call
the real DiT or return the cached output.

When To Use

Use this when you need predictable skip ratios. Because it ignores input
similarity, it should be treated as an ablation baseline rather than a final
quality method.
