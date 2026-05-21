# Headless Flux Efficient Inference Harness

This checkout is stripped down for code-only experimentation. The frontend,
HTTP server, prompt queue, history polling path, and full ComfyUI extra-node
autoload are not part of the benchmark. Workflows are executed by a small
in-process Python runner that directly calls the model/loading/sampling
functions needed by the canvas.

The useful experiment surface is:

- `efficient_skill/`: independent efficient-inference skill functions.
- `model/`: Flux model metadata and download helpers.
- `draft_canvas/`: prototype workflow canvases, including
  `flux_schnell_cache_canvas.py`.
- `benchmark/`: fixed prompt/seed benchmark and image metrics.
- `README_REVIEWER.md`: read-only planning agent contract.
- `README_EXECUTOR.md`: implementation agent contract.
- `README_CHECKER.md`: benchmark integrity and measurement agent contract.
- `agent_loop/`: JSON templates plus wrappers for checker-result conversion
  and dirty-tree diff snapshots.
- `models/diffusion_models/`: local split Flux DiT/diffusion-model storage.
- `models/text_encoders/`: local split Flux text encoder storage.
- `models/vae/`: local split Flux VAE storage.

Run the demo:

```bash
conda activate auto_deploy_flux_eff
python model/download_flux_schnell.py
python benchmark/flux_schnell_benchmark.py --no-download
```

The benchmark writes baseline images, cache-variant images, timing, and
image-difference metrics under
`benchmark/artifacts/flux_schnell_cache_benchmark/`.
