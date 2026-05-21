from __future__ import annotations

from typing import Callable

import torch
from torch import nn


_MASKED_REASON = (
    "nunchaku_extracted_linear is masked by request. "
    "The extracted Nunchaku Linear layer is retained in source but is not an active Flux quantization skill."
)


_DUPLICATE_FAKE_REGISTRATION_PATCHED = False
_KERNEL_FEATURE_ALIGNMENT = 128


def _ceil_divide(a: int, b: int) -> int:
    return (a + b - 1) // b


def _patch_duplicate_fake_registration() -> None:
    global _DUPLICATE_FAKE_REGISTRATION_PATCHED
    if _DUPLICATE_FAKE_REGISTRATION_PATCHED:
        return
    original_register_fake = torch.library.register_fake

    def safe_register_fake(op, func=None, /, **kwargs):
        decorator = original_register_fake(op, **kwargs)

        def wrap(fake_func):
            try:
                return decorator(fake_func)
            except RuntimeError as exc:
                if "already has an DispatchKey::Meta implementation" in str(exc):
                    return fake_func
                raise

        if func is None:
            return wrap
        return wrap(func)

    torch.library.register_fake = safe_register_fake
    _DUPLICATE_FAKE_REGISTRATION_PATCHED = True


def _load_nunchaku_ops() -> tuple[Callable, Callable]:
    _patch_duplicate_fake_registration()
    try:
        from nunchaku.ops.gemm import svdq_gemm_w4a4_cuda
        from nunchaku.ops.quantize import svdq_quantize_w4a4_act_fuse_lora_cuda
    except Exception as exc:
        raise RuntimeError(
            "NunchakuSVDQW4A4Linear requires the Nunchaku CUDA extension. "
            "This file only extracts the Python Linear layer; the W4A4 kernel is still "
            "implemented behind nunchaku._C.ops.gemm_w4a4 and "
            "nunchaku._C.ops.quantize_w4a4_act_fuse_lora."
        ) from exc
    return svdq_quantize_w4a4_act_fuse_lora_cuda, svdq_gemm_w4a4_cuda


def _load_nunchaku_weight_packer():
    _patch_duplicate_fake_registration()
    try:
        from nunchaku.lora.flux.packer import NunchakuWeightPacker
    except Exception as exc:
        raise RuntimeError(
            "Local Nunchaku weight conversion requires nunchaku.lora.flux.packer. "
            "The fused forward path only needs nunchaku._C.ops, but converting a "
            "plain torch.nn.Linear weight into qweight/wscales needs the Python "
            "packing utilities from the Nunchaku source tree."
        ) from exc
    return NunchakuWeightPacker


def _validate_kernel_shape(in_features: int, out_features: int, rank: int, precision: str) -> None:
    if in_features % _KERNEL_FEATURE_ALIGNMENT != 0:
        raise ValueError(
            "Nunchaku W4A4 kernels expect in_features to be a multiple of "
            f"{_KERNEL_FEATURE_ALIGNMENT}; got {in_features}."
        )
    if out_features % _KERNEL_FEATURE_ALIGNMENT != 0:
        raise ValueError(
            "Nunchaku W4A4 kernels expect out_features to be a multiple of "
            f"{_KERNEL_FEATURE_ALIGNMENT}; got {out_features}."
        )
    if rank <= 0 or rank % 16 != 0:
        raise ValueError(
            "Nunchaku W4A4 extracted Linear requires a positive low-rank dimension "
            f"that is a multiple of 16; got {rank}."
        )
    group_size = 16 if precision == "nvfp4" else 64
    if in_features % group_size != 0:
        raise ValueError(
            f"Nunchaku {precision} scales expect in_features to be divisible by {group_size}; "
            f"got {in_features}."
        )


class NunchakuSVDQW4A4Linear(nn.Module):
    """Source-extracted Nunchaku SVDQuant Linear shell.

    This mirrors `nunchaku.models.linear.SVDQW4A4Linear` without importing the
    full Nunchaku model wrapper. The actual fast path still depends on the two
    Nunchaku CUDA ops used by the original layer:

    - `svdq_quantize_w4a4_act_fuse_lora_cuda`
    - `svdq_gemm_w4a4_cuda`

    Keeping this layer separate is useful for experiments that want to replace
    only selected `nn.Linear` modules. It is not a pure-Python kernel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        act_unsigned: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        if precision == "fp4":
            precision = "nvfp4"
        if precision not in {"int4", "nvfp4"}:
            raise ValueError(f"Invalid precision: {precision}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.precision = precision
        self.torch_dtype = torch_dtype
        self.group_size = 16 if precision == "nvfp4" else 64
        self.act_unsigned = bool(act_unsigned)

        _validate_kernel_shape(self.in_features, self.out_features, self.rank, self.precision)

        self.qweight = nn.Parameter(
            torch.empty(self.out_features, self.in_features // 2, dtype=torch.int8, device=device),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(torch.empty(self.out_features, dtype=torch_dtype, device=device), requires_grad=True)
            if bias
            else None
        )
        self.wscales = nn.Parameter(
            torch.empty(
                self.in_features // self.group_size,
                self.out_features,
                dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        self.smooth_factor = nn.Parameter(
            torch.empty(self.in_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.smooth_factor_orig = nn.Parameter(
            torch.empty(self.in_features, dtype=torch_dtype, device=device), requires_grad=False
        )
        self.proj_down = nn.Parameter(torch.empty(self.in_features, self.rank, dtype=torch_dtype, device=device))
        self.proj_up = nn.Parameter(torch.empty(self.out_features, self.rank, dtype=torch_dtype, device=device))

        if precision == "nvfp4":
            self.wcscales = nn.Parameter(
                torch.ones(self.out_features, dtype=torch_dtype, device=device), requires_grad=False
            )
            self.wtscale = 1.0
        else:
            self.wcscales = None
            self.wtscale = None

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs) -> "NunchakuSVDQW4A4Linear":
        in_features = kwargs.pop("in_features", linear.in_features)
        torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)
        return cls(
            in_features=in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            torch_dtype=torch_dtype,
            device=linear.weight.device,
            **kwargs,
        )

    @classmethod
    def from_linear_quantized(
        cls,
        linear: nn.Linear,
        rank: int = 16,
        precision: str = "int4",
        svd_niter: int = 1,
        **kwargs,
    ) -> "NunchakuSVDQW4A4Linear":
        layer = cls.from_linear(linear, rank=rank, precision=precision, **kwargs)
        layer.quantize_from_linear(linear, svd_niter=svd_niter)
        return layer

    def quantize_from_linear(self, linear: nn.Linear, svd_niter: int = 1) -> None:
        """Initialize qweight/wscales/low-rank tensors from a plain Linear.

        This is a local PTQ adapter for experiments. It uses Nunchaku's real
        packing layout and real fused W4A4 forward kernel, but it is not the
        full paper pipeline from DeepCompressor because it does not calibrate
        SVDQuant smoothing factors from activations.
        """
        if self.precision != "int4":
            raise NotImplementedError(
                "Local conversion from torch.nn.Linear is currently implemented for INT4 only. "
                "Nunchaku NVFP4 should be loaded from a calibrated Nunchaku/DeepCompressor checkpoint."
            )
        if linear.in_features != self.in_features or linear.out_features != self.out_features:
            raise ValueError(
                "Linear shape mismatch: "
                f"source=({linear.out_features}, {linear.in_features}) "
                f"target=({self.out_features}, {self.in_features})."
            )

        packer_cls = _load_nunchaku_weight_packer()
        packer = packer_cls(bits=4)
        dtype = self.torch_dtype
        device = self.qweight.device
        weight = linear.weight.detach().to(device=device, dtype=torch.float32)
        rank = max(0, min(self.rank, self.in_features, self.out_features))

        if rank > 0:
            q = min(rank + 8, min(weight.shape))
            u, s, v = torch.svd_lowrank(weight, q=q, niter=max(0, int(svd_niter)))
            s_root = torch.sqrt(s[:rank])
            proj_up = u[:, :rank] * s_root.view(1, -1)
            proj_down = v[:, :rank] * s_root.view(1, -1)
            low_rank = proj_up @ proj_down.T
        else:
            proj_up = weight.new_zeros(self.out_features, 0)
            proj_down = weight.new_zeros(self.in_features, 0)
            low_rank = weight.new_zeros(weight.shape)

        residual = weight - low_rank
        quant_weight = torch.empty_like(residual, dtype=torch.int32)
        scales = torch.empty(
            self.out_features,
            self.in_features // self.group_size,
            dtype=torch.float32,
            device=device,
        )
        for group_idx, start in enumerate(range(0, self.in_features, self.group_size)):
            end = start + self.group_size
            block = residual[:, start:end]
            scale = block.abs().amax(dim=1).clamp_min(1e-8) / 7.0
            quant_weight[:, start:end] = torch.round(block / scale[:, None]).clamp(-7, 7).to(torch.int32)
            scales[:, group_idx] = scale

        packed_qweight = packer.pack_weight(quant_weight)
        packed_scales = packer.pack_scale(scales.to(dtype=dtype), group_size=self.group_size)
        packed_down = packer.pack_lowrank_weight(proj_down.T.contiguous().to(dtype=dtype), down=True)
        packed_up = packer.pack_lowrank_weight(proj_up.contiguous().to(dtype=dtype), down=False)

        with torch.no_grad():
            self.qweight.copy_(packed_qweight.to(device=device, dtype=self.qweight.dtype))
            self.wscales.copy_(packed_scales.to(device=device, dtype=self.wscales.dtype))
            self.smooth_factor.fill_(1)
            self.smooth_factor_orig.fill_(1)
            self.proj_down.copy_(packed_down.to(device=device, dtype=self.proj_down.dtype))
            self.proj_up.copy_(packed_up.to(device=device, dtype=self.proj_up.dtype))
            if self.bias is not None:
                if linear.bias is None:
                    self.bias.zero_()
                else:
                    self.bias.copy_(linear.bias.detach().to(device=device, dtype=self.bias.dtype))

    @property
    def has_nunchaku_kernel(self) -> bool:
        try:
            _load_nunchaku_ops()
        except RuntimeError:
            return False
        return True

    def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim < 2:
            raise ValueError(f"Expected input with at least 2 dims, got shape {tuple(x.shape)}.")
        original_shape = x.shape[:-1]
        channels = x.shape[-1]
        if channels != self.in_features:
            raise ValueError(f"Expected input channels {self.in_features}, got {channels}.")
        x_2d = x.reshape(-1, channels)
        if output is None:
            output = torch.empty(x_2d.shape[0], self.out_features, dtype=x.dtype, device=x.device)
        quantized_x, ascales, lora_act_out = self.quantize(x_2d)
        output = self.forward_quant(quantized_x, ascales, lora_act_out, output)
        return output.reshape(*original_shape, self.out_features)

    def quantize(self, x: torch.Tensor, pad_size: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantize_op, _ = _load_nunchaku_ops()
        batch_size, channels = x.shape
        batch_size_pad = _ceil_divide(batch_size, pad_size) * pad_size
        quantized_x = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=x.device)
        if self.precision == "nvfp4":
            ascales = torch.empty(
                channels // 16, batch_size_pad, dtype=torch.float8_e4m3fn, device=x.device
            )
        else:
            ascales = torch.empty(channels // 64, batch_size_pad, dtype=x.dtype, device=x.device)
        lora_act_out = torch.empty(batch_size_pad, self.rank, dtype=torch.float32, device=x.device)
        quantize_op(
            x,
            output=quantized_x,
            oscales=ascales,
            lora_down=self.proj_down,
            lora_act_out=lora_act_out,
            smooth=self.smooth_factor,
            fp4=self.precision == "nvfp4",
            pad_size=pad_size,
        )
        return quantized_x, ascales, lora_act_out

    def forward_quant(
        self,
        quantized_x: torch.Tensor,
        ascales: torch.Tensor,
        lora_act: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, gemm_op = _load_nunchaku_ops()
        if output is None:
            output = torch.empty(
                quantized_x.shape[0], self.out_features, dtype=self.proj_up.dtype, device=quantized_x.device
            )
        gemm_op(
            act=quantized_x,
            wgt=self.qweight,
            out=output,
            ascales=ascales,
            wscales=self.wscales,
            lora_act_in=lora_act,
            lora_up=self.proj_up,
            bias=self.bias,
            fp4=self.precision == "nvfp4",
            alpha=self.wtscale,
            wcscales=self.wcscales,
            act_unsigned=self.act_unsigned,
        )
        return output

    def __repr__(self) -> str:
        return (
            f"NunchakuSVDQW4A4Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, rank={self.rank}, "
            f"precision={self.precision}, act_unsigned={self.act_unsigned})"
        )


def nunchaku_svdq_linear_layer_spec(
    precision: str = "nvfp4",
    rank: int = 32,
    requires_kernel: bool = True,
) -> dict:
    raise RuntimeError(_MASKED_REASON)
    precision = "nvfp4" if precision == "fp4" else precision
    if precision not in {"int4", "nvfp4"}:
        raise ValueError(f"Unsupported Nunchaku SVDQuant precision: {precision}")
    return {
        "backend": "nunchaku_extracted_linear",
        "layer_class": "NunchakuSVDQW4A4Linear",
        "precision": precision,
        "rank": rank,
        "requires": ["nunchaku._C.ops"] if requires_kernel else [],
        "kernel_ops": [
            "nunchaku._C.ops.quantize_w4a4_act_fuse_lora",
            "nunchaku._C.ops.gemm_w4a4",
        ],
    }
