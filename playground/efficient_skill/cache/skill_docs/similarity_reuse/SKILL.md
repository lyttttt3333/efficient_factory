# Similarity Reuse Cache

Status: active Flux skill.

Group: `cache`

Implementation:

- Skill helper: `efficient_skill.cache.similarity_reuse.insert_similarity_reuse_cache`
- Runtime node: `SimilarityReuseCache`
- Flux canvas: `draft_canvas.flux_schnell_cache_canvas.build_flux_schnell_similarity_reuse_demo`
- Benchmark id: `similarity_reuse` in `benchmark.flux_schnell_benchmark.CACHE_BUILDERS`

What It Changes

Similarity reuse skips a DiT call when the relative change between the current
latent input and the previous latent input is below `similarity_threshold`.
The comparison can be spatially subsampled with `subsample_factor`.

How To Embed

Patch the DiT model output of the split loader.

```python
from efficient_skill.cache.similarity_reuse import insert_similarity_reuse_cache
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_similarity_reuse_cache(
    workflow,
    model_ref=output_ref("1", 0),
    similarity_threshold=1.0,
    subsample_factor=8,
    warmup_steps=1,
    max_skip_steps=1,
)
```

Direct Model Integration

Attach a reuse holder with mode `similarity` to the DiT model options and wrap
predict-noise. Store the previous latent input and cached output inside the
holder. Reset state at the end of each sample.

When To Use

Use this when fixed periodic skipping is too blunt. Sweep
`similarity_threshold` and `subsample_factor` with fixed prompt/noise cases.
