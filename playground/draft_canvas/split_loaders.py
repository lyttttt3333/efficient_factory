from __future__ import annotations


def diffusion_model_loader_node(unet_name: str, weight_dtype: str = "default") -> dict:
    return {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": unet_name,
            "weight_dtype": weight_dtype,
        },
    }


def dual_clip_loader_node(
    clip_name1: str,
    clip_name2: str,
    clip_type: str = "flux",
    device: str = "default",
) -> dict:
    return {
        "class_type": "DualCLIPLoader",
        "inputs": {
            "clip_name1": clip_name1,
            "clip_name2": clip_name2,
            "type": clip_type,
            "device": device,
        },
    }


def vae_loader_node(vae_name: str) -> dict:
    return {
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": vae_name,
        },
    }
