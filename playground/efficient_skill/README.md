# Efficient Skill

This package contains only reusable efficient-inference skill functions. These
helpers are intentionally independent so other agents can compose them in
`draft_canvas/` without coupling one skill to another.

The active per-skill reading cards are indexed in `SKILL_INDEX.md`.

Implemented skill groups:

- `cache/`: cache-style model patches: EasyCache, LazyCache, Flux TeaCache,
  periodic reuse, input-similarity reuse, delta reuse, and EMA reuse.
- `compile/`: `torch.compile` workflow patch helpers.
- `quantization/`: quantization method catalog plus active independent
  adapters for TorchAO FP8/MXFP8, TorchAO NVFP4, and selective per-shape
  Linear quantization that first benchmarks quantized vs high-precision Linear
  at the actual runtime shape and only quantizes shapes that pass the
  configured speedup threshold.
- `sparse_attention/`: official GPU-kernel sparse attention wrappers only.
  Flux canvas/benchmark exposes PISA and SpargeAttn. Sparse VideoGen and
  Sparse VideoGen2 remain as video-kernel backend specs, not Flux image DiT
  choices. Prototype-only methods were removed.

Support helpers:

- `attention/`: attention backend launch flag helpers. This is launch
  configuration; sparse-attention implementations live under
  `sparse_attention/`.
- `common/`: workflow and image/API utilities shared by skills and benchmark.

Quantization notes:

- TorchAO adapters are local workflow model patches through
  `TorchAOQuantizeModel`. They require `torchao` in the active conda env.
- TensorRT ModelOpt, standalone SVDQuant Linear, extracted Nunchaku Linear, and
  Nunchaku/SVDQuant backend specs are masked for now. Their source is retained,
  but they are not exposed through the active Flux canvas/benchmark path.
