# SpargeAttn Sparse Attention

Status: active Flux skill.

Group: `sparse_attention`

Implementation:

- Skill helper: `efficient_skill.sparse_attention.insert_spargeattn_sparse_attention`
- Runtime node: `SparseAttentionModel`
- Official kernel entrypoint: `spas_sage_attn.spas_sage2_attn_meansim_topk_cuda`
- Flux canvas: `draft_canvas.flux_schnell_sparse_attention_canvas.build_flux_schnell_spargeattn_sparse_attention_demo`
- Benchmark id: `spargeattn`

What It Changes

SpargeAttn replaces eligible Flux DiT attention calls with the official
SpargeAttn GPU kernel. The default Flux configuration applies only to
single-stream blocks and uses `topk=0.25`.

How To Embed

Insert after the split diffusion model loader.

```python
from efficient_skill.sparse_attention import insert_spargeattn_sparse_attention
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_spargeattn_sparse_attention(
    workflow,
    model_ref=output_ref("1", 0),
    apply_to="single",
    min_tokens=128,
    max_tokens=1_000_000,
    topk=0.25,
)
```

The helper adds a `SparseAttentionModel` patch node and retargets downstream
model consumers to the sparse-attention patched model.

Direct Model Integration

For raw Python inference, intercept the DiT attention backend and call
`spas_sage2_attn_meansim_topk_cuda` on CUDA Q/K/V tensors in HND layout. Keep a
dense fallback for unsupported masks, cross-attention-like token mismatches, or
unsupported head dimensions.

When To Use

Use this for official-kernel sparse attention tests on long token sequences.
Benchmark with warmup and inspect `official_kernel_calls` vs dense passthroughs.
