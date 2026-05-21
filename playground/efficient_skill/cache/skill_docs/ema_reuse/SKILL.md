# EMA Reuse Cache

Status: active Flux skill.

Group: `cache`

Implementation:

- Skill helper: `efficient_skill.cache.ema_reuse.insert_ema_reuse_cache`
- Runtime node: `EMAReuseCache`
- Flux canvas: `draft_canvas.flux_schnell_cache_canvas.build_flux_schnell_ema_reuse_demo`
- Benchmark id: `ema_reuse` in `benchmark.flux_schnell_benchmark.CACHE_BUILDERS`

What It Changes

EMA reuse skips scheduled DiT calls and returns an exponential moving average of
recent denoise outputs. It smooths the reused output instead of returning only
the most recent output.

How To Embed

Patch the split DiT model before it is consumed by the guider.

```python
from efficient_skill.cache.ema_reuse import insert_ema_reuse_cache
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_ema_reuse_cache(
    workflow,
    model_ref=output_ref("1", 0),
    ema_decay=0.5,
    interval=2,
    warmup_steps=1,
    max_skip_steps=1,
)
```

Direct Model Integration

Maintain `ema_output` in the denoise-loop holder. Update it after real DiT
calls and return it on skipped calls. Reset it for each new sample.

When To Use

Use this when delta extrapolation is unstable but plain output reuse is too
noisy. Benchmark it against `periodic_reuse` at the same skip schedule.
