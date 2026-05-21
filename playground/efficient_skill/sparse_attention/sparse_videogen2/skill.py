from __future__ import annotations

from typing import Any


def sparse_videogen2_official_backend_spec(**kwargs) -> dict[str, Any]:
    return {
        "backend": "official_video_gpu_kernel",
        "method": "sparse_videogen2",
        "package": "svg",
        "entrypoints": [
            "svg.kmeans_utils.dynamic_block_sparse_fwd_flashinfer",
            "svg.kmeans_utils.dynamic_block_sparse_fwd_triton",
        ],
        "source": "https://github.com/svg-project/Sparse-VideoGen",
        "flux_canvas": False,
        "notes": (
            "Official dynamic block sparse kernels need the Sparse VideoGen video "
            "block map pipeline, so this is kept as a backend spec for future video canvases."
        ),
        "inputs": kwargs,
    }
