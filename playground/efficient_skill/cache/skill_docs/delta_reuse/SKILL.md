# Delta Reuse Cache

Status: active Flux skill.

Group: `cache`

Implementation:

- Skill helper: `efficient_skill.cache.delta_reuse.insert_delta_reuse_cache`
- Runtime node: `DeltaReuseCache`
- Flux canvas: `draft_canvas.flux_schnell_cache_canvas.build_flux_schnell_delta_reuse_demo`
- Benchmark id: `delta_reuse` in `benchmark.flux_schnell_benchmark.CACHE_BUILDERS`

What It Changes

Delta reuse skips scheduled DiT calls and predicts the skipped output from the
last output plus a scaled difference from the previous output. This can preserve
some denoise trend compared with returning the last output unchanged.

How To Embed

Insert the patch node after the split DiT loader.

```python
from efficient_skill.cache.delta_reuse import insert_delta_reuse_cache
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_delta_reuse_cache(
    workflow,
    model_ref=output_ref("1", 0),
    delta_scale=1.0,
    interval=2,
    warmup_steps=2,
    max_skip_steps=1,
)
```

Direct Model Integration

Store `previous_output` and `last_output` in the denoise-loop holder. On a skip,
return `last_output + delta_scale * (last_output - previous_output)` instead of
executing the DiT.

When To Use

Use this to test whether cheap extrapolation gives better quality than plain
periodic reuse at the same skipped-step count.
