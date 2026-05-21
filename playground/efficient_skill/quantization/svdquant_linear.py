from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from efficient_skill.common.workflow import Workflow, next_node_id, output_ref, replace_model_input
from efficient_skill.quantization.torchao_fp8_mxfp8 import DEFAULT_FLUX_SKIP_MODULES


_MASKED_REASON = (
    "standalone_svdquant_linear is masked by request. "
    "The pure-PyTorch reference layer is retained in source but is not an active Flux quantization skill."
)


@dataclass(frozen=True)
class SVDQuantConfig:
    rank: int = 32
    group_size: int = 64
    cache_dequant_weight: bool = True
    svd_niter: int = 4


def _pack_signed_int4(q: torch.Tensor) -> torch.Tensor:
    q = torch.clamp(q.to(torch.int16), -8, 7) + 8
    if q.shape[-1] % 2:
        q = F.pad(q, (0, 1), value=8)
    lo = q[..., 0::2]
    hi = q[..., 1::2]
    return ((hi << 4) | lo).to(torch.uint8)


def _unpack_signed_int4(packed: torch.Tensor, original_cols: int) -> torch.Tensor:
    packed = packed.to(torch.int16)
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    q = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.int16, device=packed.device)
    q[..., 0::2] = lo
    q[..., 1::2] = hi
    q = q[..., :original_cols] - 8
    return q.to(torch.float32)


class StandaloneSVDQuantLinear(torch.nn.Module):
    """Pure-PyTorch SVDQuant-style Linear.

    This intentionally does not call Nunchaku's C++/CUDA wrappers. It stores a
    signed INT4 residual branch plus a BF16/FP16 low-rank branch:

        y = linear(x, dequant_int4_residual) + (x @ proj_down) @ proj_up.T + bias

    It is useful for local algorithm wiring and quality checks. It is not the
    fused W4A4 Nunchaku kernel and should not be expected to match its speed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        group_size: int = 64,
        bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        cache_dequant_weight: bool = True,
        svd_niter: int = 4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(0, min(int(rank), in_features, out_features))
        self.group_size = max(1, int(group_size))
        self.cache_dequant_weight = cache_dequant_weight
        self.svd_niter = max(0, int(svd_niter))
        self.register_buffer("qweight", torch.empty(out_features, math.ceil(in_features / 2), dtype=torch.uint8, device=device))
        self.register_buffer("wscales", torch.empty(math.ceil(in_features / group_size), out_features, dtype=dtype, device=device))
        self.proj_down = torch.nn.Parameter(torch.empty(in_features, self.rank, dtype=dtype, device=device), requires_grad=False)
        self.proj_up = torch.nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device), requires_grad=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, dtype=dtype, device=device), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self._cached_weight: torch.Tensor | None = None

    @classmethod
    def from_linear(
        cls,
        linear: torch.nn.Linear,
        rank: int = 32,
        group_size: int = 64,
        cache_dequant_weight: bool = True,
        svd_niter: int = 4,
    ) -> "StandaloneSVDQuantLinear":
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            group_size=group_size,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
            cache_dequant_weight=cache_dequant_weight,
            svd_niter=svd_niter,
        )
        layer.quantize_from_linear(linear)
        return layer

    def quantize_from_linear(self, linear: torch.nn.Linear) -> None:
        weight = linear.weight.detach().float()
        rank = self.rank
        if rank > 0:
            # Full SVD for every Flux Linear is prohibitively expensive. The
            # low-rank solver keeps this standalone layer practical while still
            # preserving the SVDQuant split: low-rank outlier branch + 4-bit
            # residual branch.
            q = min(rank + 8, min(weight.shape))
            u, s, v = torch.svd_lowrank(weight, q=q, niter=self.svd_niter)
            s_root = torch.sqrt(s[:rank])
            proj_up = u[:, :rank] * s_root.view(1, -1)
            proj_down = v[:, :rank] * s_root.view(1, -1)
            low_rank = proj_up @ proj_down.T
        else:
            proj_up = weight.new_zeros(weight.shape[0], 0)
            proj_down = weight.new_zeros(weight.shape[1], 0)
            low_rank = weight.new_zeros(weight.shape)

        residual = weight - low_rank
        qweight = torch.empty_like(residual, dtype=torch.int16)
        scales = torch.empty_like(self.wscales, dtype=torch.float32)
        for group_idx, start in enumerate(range(0, self.in_features, self.group_size)):
            end = min(start + self.group_size, self.in_features)
            block = residual[:, start:end]
            scale = block.abs().amax(dim=1).clamp_min(1e-8) / 7.0
            qweight[:, start:end] = torch.round(block / scale[:, None]).clamp(-8, 7).to(torch.int16)
            scales[group_idx] = scale

        self.qweight.copy_(_pack_signed_int4(qweight).to(self.qweight.device))
        self.wscales.copy_(scales.to(device=self.wscales.device, dtype=self.wscales.dtype))
        self.proj_up.copy_(proj_up.to(device=self.proj_up.device, dtype=self.proj_up.dtype))
        self.proj_down.copy_(proj_down.to(device=self.proj_down.device, dtype=self.proj_down.dtype))
        if self.bias is not None and linear.bias is not None:
            self.bias.copy_(linear.bias.detach().to(device=self.bias.device, dtype=self.bias.dtype))
        self._cached_weight = None

    def dequant_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.cache_dequant_weight and self._cached_weight is not None:
            if self._cached_weight.device == device and self._cached_weight.dtype == dtype:
                return self._cached_weight
        q = _unpack_signed_int4(self.qweight.to(device), self.in_features)
        weight = torch.empty(self.out_features, self.in_features, dtype=torch.float32, device=device)
        scales = self.wscales.to(device=device, dtype=torch.float32)
        for group_idx, start in enumerate(range(0, self.in_features, self.group_size)):
            end = min(start + self.group_size, self.in_features)
            weight[:, start:end] = q[:, start:end] * scales[group_idx].view(-1, 1)
        weight = weight.to(dtype=dtype)
        if self.cache_dequant_weight:
            self._cached_weight = weight
        return weight

    def _apply(self, fn):
        self._cached_weight = None
        return super()._apply(fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        residual_weight = self.dequant_weight(dtype=x2.dtype, device=x2.device)
        bias = self.bias.to(device=x2.device, dtype=x2.dtype) if self.bias is not None else None
        out = F.linear(x2, residual_weight, bias)
        if self.rank > 0:
            down = self.proj_down.to(device=x2.device, dtype=x2.dtype)
            up = self.proj_up.to(device=x2.device, dtype=x2.dtype)
            out = out + (x2 @ down) @ up.T
        return out.reshape(*original_shape, self.out_features)


def svdquant_linear_node(
    model_ref: list,
    rank: int = 32,
    group_size: int = 64,
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
    max_modules: int = 0,
    cache_dequant_weight: bool = True,
    svd_niter: int = 4,
    cache_quantized_model: bool = True,
) -> dict:
    raise RuntimeError(_MASKED_REASON)
    return {
        "class_type": "StandaloneSVDQuantLinearModel",
        "inputs": {
            "model": model_ref,
            "rank": rank,
            "group_size": group_size,
            "skip_modules": skip_modules,
            "max_modules": max_modules,
            "cache_dequant_weight": cache_dequant_weight,
            "svd_niter": svd_niter,
            "cache_quantized_model": cache_quantized_model,
        },
    }


def insert_standalone_svdquant_linear(
    workflow: Workflow,
    model_ref: list,
    rank: int = 32,
    group_size: int = 64,
    skip_modules: str = DEFAULT_FLUX_SKIP_MODULES,
    max_modules: int = 0,
    cache_dequant_weight: bool = True,
    svd_niter: int = 4,
    cache_quantized_model: bool = True,
) -> tuple[Workflow, list]:
    raise RuntimeError(_MASKED_REASON)
    node_id = next_node_id(workflow)
    workflow[node_id] = svdquant_linear_node(
        model_ref=model_ref,
        rank=rank,
        group_size=group_size,
        skip_modules=skip_modules,
        max_modules=max_modules,
        cache_dequant_weight=cache_dequant_weight,
        svd_niter=svd_niter,
        cache_quantized_model=cache_quantized_model,
    )
    new_model_ref = output_ref(node_id, 0)
    replace_model_input(workflow, old_ref=model_ref, new_ref=new_model_ref)
    workflow[node_id]["inputs"]["model"] = model_ref
    return workflow, new_model_ref
