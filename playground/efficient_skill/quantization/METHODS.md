# Quantization Method Catalog

This folder is the quantization skill space for local Flux inference. The goal is
to keep each quantization method as an independent adapter that other agents can
compose in `draft_canvas/` and measure in `benchmark/`.

## Real Backend Targets

| id | method | target precision | integration status | notes |
|---|---|---|---|---|
| `torchao_fp8_dynamic` | TorchAO dynamic FP8 activation + FP8 weight | FP8 W8A8 | merged as a local model patch node | PyTorch-native path; best first smoke test when TorchAO is installed. |
| `torchao_mxfp8_dynamic` | TorchAO MXFP8 dynamic activation + MXFP8 weight | MXFP8 W8A8 | merged as a local model patch node | Blackwell microscaling FP8 path; expected to be close to BF16 quality. |
| `torchao_nvfp4_dynamic` | TorchAO NVFP4 dynamic activation + NVFP4 weight | NVFP4 W4A4 | merged as a local model patch node | Blackwell-only path; highest speed/memory upside, larger quality risk than MXFP8. |
| `selective_torchao_*` | Runtime per-shape selective TorchAO quantization | FP8/MXFP8/NVFP4 | merged as a local model patch node | Benchmarks each actual Linear runtime shape against high precision, then quantizes only shapes whose isolated Linear speedup passes `min_speedup`. |

## Masked / Temporarily Disabled

These entries are kept in source for later recovery, but they are not exported
from `efficient_skill.quantization`, not listed in the Flux quantization canvas,
and their public entrypoint functions raise `RuntimeError`.

| id | method | reason masked |
|---|---|---|
| `modelopt_tensorrt_fp8_fp4` | TensorRT ModelOpt QDQ export/build | External engine path; no local Flux runner/export benchmark is active. |
| `standalone_svdquant_linear` | Pure-PyTorch SVDQuant-style Linear | Reference implementation, not a proved acceleration path. |
| `nunchaku_extracted_linear` | Extracted Nunchaku W4A4 Linear layer | Kernel-facing layer exists, but no Flux module selection produced a real speedup. |
| `nunchaku_svdquant_backend_spec` | Nunchaku/SVDQuant checkpoint backend spec | External checkpoint/backend path; not part of the active local Flux skill set. |

## Diffusion-Specific Quantization Work

| method | scope | key idea | how to use here |
|---|---|---|---|
| PTQ4DM | U-Net diffusion PTQ | timestep-aware calibration for changing denoise distributions | calibration/fake-quant reference for older diffusion backbones |
| Q-Diffusion | U-Net diffusion PTQ | timestep-aware calibration and split shortcut quantization | reference for shortcut/outlier handling |
| PTQD | diffusion PTQ correction | quantization noise accumulates through the sampling schedule | quality-loss analysis and sampler correction ideas |
| PTQ4DiT | DiT PTQ | channel salience balancing and timestep salience calibration | Flux-style DiT calibration policy |
| ViDiT-Q | image/video DiT PTQ | fine-grained grouping, dynamic quantization, static-dynamic channel balancing, CUDA kernels | candidate for future Flux block/kernel adaptation |
| Q-DiT | DiT PTQ | automatic granularity allocation and sample-wise dynamic activation quantization | candidate for layer/group sensitivity search |
| MixDQ | few-step T2I quantization | metric-decoupled mixed precision for few-step text-to-image diffusion | useful for Schnell-style few-step settings |
| MSFP/TALoRA | diffusion FP4 | mixup-sign FP4 quantization plus timestep-aware LoRA fine-tuning | future FP4 quality recovery route |
| QuaRTZ | diffusion 4-bit PTQ | residual truncation and zero suppression to preserve small texture activations | future 4-bit PTQ route |
| ConvRot | DiT W4A4 | rotation-based outlier suppression and fused 4-bit linear module | future plug-in W4A4 Linear route |
| Q-Drift | quantized sampling correction | drift correction over quantized diffusion sampling | sampler-side correction that can compose with quant backends |

## Source Pointers

- TorchAO Blackwell diffusion: https://pytorch.org/blog/faster-diffusion-on-blackwell-mxfp8-and-nvfp4-with-diffusers-and-torchao/
- TorchAO quantized inference docs: https://docs.pytorch.org/ao/stable/workflows/inference.html
- Torch-TensorRT quantization docs: https://docs.pytorch.org/TensorRT/user_guide/shapes_precision/quantization.html
- NVIDIA FP4 image generation: https://developer.nvidia.com/blog/?p=99256
- Transformer Engine FP8/NVFP4 notes: https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html
- PTQ4DM: https://arxiv.org/abs/2211.15736
- Q-Diffusion: https://arxiv.org/abs/2302.04304
- PTQD: https://arxiv.org/abs/2305.10657
- PTQ4DiT: https://arxiv.org/abs/2405.16005
- ViDiT-Q: https://arxiv.org/abs/2406.02540
- Q-DiT: https://arxiv.org/abs/2406.17343
- MixDQ: https://arxiv.org/abs/2405.17873
- QuaRTZ: https://arxiv.org/abs/2509.26436
- ConvRot: https://arxiv.org/abs/2512.03673
- Q-Drift: https://arxiv.org/abs/2603.18095
