from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.flux_schnell import FLUX_SCHNELL_SPLIT, FluxModelFile


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def download_component(repo_root: Path, component: FluxModelFile, force: bool = False) -> Path:
    target = component.local_path(repo_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        return target

    cached = Path(
        hf_hub_download(
            repo_id=component.repo_id,
            filename=component.filename,
            revision=component.revision,
            repo_type="model",
            local_dir=target.parent,
            force_download=force,
        )
    )
    if cached != target and cached.exists():
        shutil.copy2(cached, target)
    return target


def download_flux_schnell_split(repo_root: Path, force: bool = False) -> dict[str, Path]:
    return {
        component.role: download_component(repo_root, component, force=force)
        for component in FLUX_SCHNELL_SPLIT.components()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    targets = download_flux_schnell_split(repo_root_from_here(), force=args.force)
    for role, target in targets.items():
        print(f"{role}: {target}")


if __name__ == "__main__":
    main()
