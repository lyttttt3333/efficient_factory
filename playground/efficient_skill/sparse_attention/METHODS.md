# Sparse Attention Skills

This folder is now official GPU-kernel-only. Prototype PyTorch mask/reuse
implementations were removed from the Flux canvas and benchmark.

| skill | folder | official kernel entrypoint | Flux canvas |
|---|---|---|---|
| PISA | `pisa/` | `piecewise_attn.piecewise_sparse_attention` from `xie-lab-ml/piecewise-sparse-attention` | yes |
| SpargeAttn | `spargeattn/` | `spas_sage_attn.spas_sage2_attn_meansim_topk_cuda` from `thu-ml/SpargeAttn` | yes |
| Sparse VideoGen | `sparse_videogen/` | `svg.kernels.ops.attention_ops.sparse_attn_forward` from `svg-project/Sparse-VideoGen` | no, video metadata required |
| Sparse VideoGen2 | `sparse_videogen2/` | `svg.kmeans_utils.dynamic_block_sparse_fwd_flashinfer` / `dynamic_block_sparse_fwd_triton` from `svg-project/Sparse-VideoGen` | no, video block-map pipeline required |

Recommended Flux defaults after warmup on this machine:

| skill | default config | reason |
|---|---|---|
| PISA | `apply_to="single"`, `density=0.15`, `block_size=128`, resolution-shaped precompile | Accelerates the single-stream Flux attention path while avoiding the slow full-model replacement path and first-call Triton autotune overhead. |
| SpargeAttn | `apply_to="single"`, `topk=0.25` | Gives a measured DiT speedup with lower image loss than replacing all attention blocks. |

Installation notes:

- PISA is installed as editable `piecewise_attn` from the official
  `xie-lab-ml/piecewise-sparse-attention` repo.
- SpargeAttn is installed as editable `spas_sage_attn` from the official
  `thu-ml/SpargeAttn` repo. On RTX 5090, the build used the env CUDA 13 nvcc
  path and added `12.0` to the upstream architecture allowlist so the official
  CUDA sources compile to `sm_120`.

Removed because no official runnable GPU kernel package was found for this
repo's Flux image attention path: `ditfastattn`, `chipmunk`, `svg_ear`,
`calibatt`, and `haste`.

The benchmark reports DiT denoise-only wall time, image loss versus baseline,
attention call counts, official kernel calls, and kernel-reported or
kernel-derived skipped attention fraction. Missing official packages raise a
runtime error instead of falling back to local prototype code.
