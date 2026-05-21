from __future__ import annotations


def launch_flags(backend: str) -> list[str]:
    """Return ComfyUI launch flags for a requested attention backend."""
    if backend == "default":
        return []
    mapping = {
        "pytorch": "--use-pytorch-cross-attention",
        "xformers_disabled": "--disable-xformers",
        "split": "--use-split-cross-attention",
        "sub_quad": "--use-quad-cross-attention",
        "sage": "--use-sage-attention",
        "flash": "--use-flash-attention",
    }
    if backend not in mapping:
        raise ValueError(f"Unknown attention backend: {backend}")
    return [mapping[backend]]

