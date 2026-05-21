# PISA Sparse Attention

Status: active Flux skill.

Group: `sparse_attention`

Implementation:

- Skill helper: `efficient_skill.sparse_attention.insert_pisa_sparse_attention`
- Runtime node: `SparseAttentionModel`
- Official kernel entrypoint: `piecewise_attn.piecewise_sparse_attention`
- Flux canvas: `draft_canvas.flux_schnell_sparse_attention_canvas.build_flux_schnell_pisa_sparse_attention_demo`
- Benchmark id: `pisa`

What It Changes

PISA replaces eligible Flux DiT attention calls with the official PISA GPU
kernel. It does not alter Linear layers, text encoders, VAE, scheduler, or
noise. By default it applies to Flux single-stream blocks only.

How To Embed

Insert after the split diffusion model loader and before the guider uses the
model.

```python
from efficient_skill.sparse_attention import insert_pisa_sparse_attention
from efficient_skill.common.workflow import output_ref

workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_pisa_sparse_attention(
    workflow,
    model_ref=output_ref("1", 0),
    apply_to="single",
    min_tokens=128,
    max_tokens=1_000_000,
    density=0.15,
    block_size=128,
)
```

The runtime patch installs an attention override in
`model_options["transformer_options"]["optimized_attention_override"]` during
sampling, then restores the original model options afterward. The Flux canvas
precompiles the official PISA kernel for the current resolution before sampling
so Triton autotuning is not charged to the DiT denoise timer.

Direct Model Integration

For a raw DiT runner, route self-attention Q/K/V tensors shaped as
`[batch, heads, tokens, head_dim]` or canonicalizable `[batch, tokens, inner]`
through `piecewise_sparse_attention`. Fall back to dense attention when tokens
are outside the configured range, masks are present, Q/K/V token lengths differ,
or head dim is unsupported.

When To Use

Use PISA when token count is large enough for sparse attention to amortize
overhead. Always warm up kernels before timing.
