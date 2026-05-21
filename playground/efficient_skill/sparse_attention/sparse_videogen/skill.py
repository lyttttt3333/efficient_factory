from __future__ import annotations

from typing import Any


def sparse_videogen_official_backend_spec(**kwargs) -> dict[str, Any]:
    return {
        "backend": "official_video_gpu_kernel",
        "method": "sparse_videogen",
        "package": "svg",
        "entrypoint": "svg.kernels.ops.attention_ops.sparse_attn_forward",
        "source": "https://github.com/svg-project/Sparse-VideoGen",
        "flux_canvas": False,
        "notes": (
            "Official Sparse VideoGen kernels require video metadata and are not "
            "a direct Flux image DiT attention drop-in."
        ),
        "inputs": kwargs,
    }
