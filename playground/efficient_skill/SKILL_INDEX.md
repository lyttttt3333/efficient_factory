# Active Efficient Skills

This index lists the skills currently exposed to agents for Flux experiments.
Masked quantization entries and video-only backend specs are intentionally not
listed here.

## Cache

- `easycache`: `efficient_skill/cache/skill_docs/easycache/SKILL.md`
- `lazycache`: `efficient_skill/cache/skill_docs/lazycache/SKILL.md`
- `teacache`: `efficient_skill/cache/skill_docs/teacache/SKILL.md`
- `periodic_reuse`: `efficient_skill/cache/skill_docs/periodic_reuse/SKILL.md`
- `similarity_reuse`: `efficient_skill/cache/skill_docs/similarity_reuse/SKILL.md`
- `delta_reuse`: `efficient_skill/cache/skill_docs/delta_reuse/SKILL.md`
- `ema_reuse`: `efficient_skill/cache/skill_docs/ema_reuse/SKILL.md`

## Quantization

- `torchao_fp8_dynamic`: `efficient_skill/quantization/skill_docs/torchao_fp8_dynamic/SKILL.md`
- `torchao_mxfp8_dynamic`: `efficient_skill/quantization/skill_docs/torchao_mxfp8_dynamic/SKILL.md`
- `torchao_nvfp4_dynamic`: `efficient_skill/quantization/skill_docs/torchao_nvfp4_dynamic/SKILL.md`
- `selective_torchao_fp8`: `efficient_skill/quantization/skill_docs/selective_torchao_fp8/SKILL.md`
- `selective_torchao_mxfp8`: `efficient_skill/quantization/skill_docs/selective_torchao_mxfp8/SKILL.md`
- `selective_torchao_nvfp4`: `efficient_skill/quantization/skill_docs/selective_torchao_nvfp4/SKILL.md`

## Sparse Attention

- `pisa`: `efficient_skill/sparse_attention/skill_docs/pisa/SKILL.md`
- `spargeattn`: `efficient_skill/sparse_attention/skill_docs/spargeattn/SKILL.md`

## Optional Helper

- `torch_compile`: `efficient_skill/compile/skill_docs/torch_compile/SKILL.md`

## Common Embedding Pattern

All active skills patch the split Flux diffusion model output, not the text
encoder or VAE. The usual canvas pattern is:

```python
workflow = build_flux_schnell_baseline(prompt=prompt, seed=seed)
insert_some_skill(workflow, model_ref=output_ref("1", 0), ...)
```

Node `1` is the split DiT loader in the current Flux canvas. The insert helper
adds a model patch node, retargets downstream model consumers to the patched
model output, and leaves the patch node input connected to the original DiT
loader output.
