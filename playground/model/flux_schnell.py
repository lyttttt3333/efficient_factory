from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FluxModelFile:
    role: str
    repo_id: str
    filename: str
    local_subdir: str
    revision: str = "main"

    def local_path(self, repo_root: Path) -> Path:
        return repo_root / self.local_subdir / self.filename


@dataclass(frozen=True)
class FluxSchnellSplitModel:
    diffusion: FluxModelFile = FluxModelFile(
        role="diffusion",
        repo_id="Comfy-Org/flux1-schnell",
        filename="flux1-schnell.safetensors",
        local_subdir="models/diffusion_models",
    )
    t5xxl: FluxModelFile = FluxModelFile(
        role="t5xxl",
        repo_id="comfyanonymous/flux_text_encoders",
        filename="t5xxl_fp8_e4m3fn.safetensors",
        local_subdir="models/text_encoders",
    )
    clip_l: FluxModelFile = FluxModelFile(
        role="clip_l",
        repo_id="comfyanonymous/flux_text_encoders",
        filename="clip_l.safetensors",
        local_subdir="models/text_encoders",
    )
    vae: FluxModelFile = FluxModelFile(
        role="vae",
        repo_id="second-state/FLUX.1-schnell-GGUF",
        filename="ae.safetensors",
        local_subdir="models/vae",
    )

    def components(self) -> tuple[FluxModelFile, ...]:
        return (self.diffusion, self.t5xxl, self.clip_l, self.vae)

    def local_paths(self, repo_root: Path) -> dict[str, Path]:
        return {component.role: component.local_path(repo_root) for component in self.components()}


FLUX_SCHNELL_SPLIT = FluxSchnellSplitModel()
