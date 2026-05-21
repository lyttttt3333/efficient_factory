from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import torch


AttentionFn = Callable[..., torch.Tensor]
FLUX_OFFICIAL_KERNEL_METHODS = ("pisa", "spargeattn")
_PISA_TRITON_ALLOCATOR_DEVICE: torch.device | None = None
_PISA_PRECOMPILED_CONFIGS: set[tuple[Any, ...]] = set()


@dataclass
class SparseAttentionConfig:
    method: str
    apply_to: str = "single"
    min_tokens: int = 128
    max_tokens: int = 1_000_000
    density: float = 0.15
    block_size: int = 64
    topk: float = 0.25
    verbose: bool = False
    precompile_tokens: int = 0
    precompile_heads: int = 24
    precompile_head_dim: int = 128

    def __post_init__(self) -> None:
        if self.method not in FLUX_OFFICIAL_KERNEL_METHODS:
            raise ValueError(
                f"Unsupported sparse attention method '{self.method}'. "
                f"Official Flux kernel methods: {', '.join(FLUX_OFFICIAL_KERNEL_METHODS)}"
            )
        if self.apply_to not in {"all", "double", "single"}:
            raise ValueError("apply_to must be one of: all, double, single")
        self.min_tokens = max(int(self.min_tokens), 1)
        self.max_tokens = max(int(self.max_tokens), self.min_tokens)
        self.block_size = max(int(self.block_size), 1)
        self.density = min(max(float(self.density), 0.0), 1.0)
        self.topk = min(max(float(self.topk), 0.0), 1.0)
        self.precompile_tokens = max(int(self.precompile_tokens), 0)
        self.precompile_heads = max(int(self.precompile_heads), 1)
        self.precompile_head_dim = max(int(self.precompile_head_dim), 1)


@dataclass
class SparseAttentionStats:
    calls: int = 0
    official_kernel_calls: int = 0
    dense_passthrough_calls: int = 0
    fallback_calls: int = 0
    attention_tokens: int = 0
    skipped_fraction_sum: float = 0.0
    skipped_fraction_count: int = 0

    def add_official_kernel(self, skipped_fraction: float | None) -> None:
        self.official_kernel_calls += 1
        if skipped_fraction is None:
            return
        self.skipped_fraction_sum += min(max(float(skipped_fraction), 0.0), 1.0)
        self.skipped_fraction_count += 1

    def snapshot(self) -> dict[str, Any]:
        skipped = (
            self.skipped_fraction_sum / self.skipped_fraction_count
            if self.skipped_fraction_count
            else 0.0
        )
        return {
            "attention_calls": self.calls,
            "official_kernel_calls": self.official_kernel_calls,
            "dense_passthrough_calls": self.dense_passthrough_calls,
            "dense_attention_calls": self.dense_passthrough_calls,
            "sparse_attention_calls": self.official_kernel_calls,
            "reused_attention_calls": 0,
            "fallback_attention_calls": self.fallback_calls,
            "attention_tokens": self.attention_tokens,
            "mean_allowed_attention_fraction": 1.0 - skipped,
            "mean_skipped_attention_fraction": skipped,
        }


@dataclass
class SparseInput:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    batch: int
    heads: int
    tokens: int
    dim: int
    skip_output_reshape: bool


class SparseAttentionHolder:
    def __init__(self, config: SparseAttentionConfig):
        self.config = config
        self.stats = SparseAttentionStats()
        self.previous_override = None
        self.last_stats: dict[str, Any] = {}
        self.denoise_step = -1
        self._pisa_kernel = None

    def clone(self) -> "SparseAttentionHolder":
        clone = SparseAttentionHolder(self.config)
        clone._pisa_kernel = self._pisa_kernel
        return clone

    def copy_stats_from(self, other: "SparseAttentionHolder") -> None:
        self.stats = other.stats
        self.last_stats = other.snapshot()

    def snapshot(self) -> dict[str, Any]:
        stats = self.stats.snapshot()
        stats["method"] = self.config.method
        stats["denoise_steps_seen"] = max(self.denoise_step + 1, 0)
        stats["backend"] = "official_gpu_kernel"
        return stats

    def attention_override(
        self,
        func: AttentionFn,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        **kwargs,
    ) -> torch.Tensor:
        transformer_options = kwargs.get("transformer_options") or {}
        self._mark_denoise_step(transformer_options)
        self.stats.calls += 1
        self.stats.attention_tokens += _attention_tokens(q, kwargs)

        if self.previous_override is not None:
            dense_func = lambda *a, **kw: self.previous_override(func, *a, **kw)
        else:
            dense_func = func

        if not self._should_try_official_kernel(q, k, v, kwargs, transformer_options):
            self.stats.dense_passthrough_calls += 1
            return _call_attention(dense_func, q, k, v, heads, kwargs)

        sparse_input = _canonicalize(q, k, v, heads, kwargs)
        if sparse_input is None or sparse_input.dim not in {64, 128}:
            self.stats.fallback_calls += 1
            return _call_attention(dense_func, q, k, v, heads, kwargs)
        if not sparse_input.q.is_cuda:
            raise RuntimeError(
                f"{self.config.method} sparse attention is configured as an official GPU kernel, "
                "but the current attention tensors are not CUDA tensors."
            )

        if self.config.method == "pisa":
            out, skipped_fraction = self._run_pisa(sparse_input, kwargs)
        elif self.config.method == "spargeattn":
            out, skipped_fraction = self._run_spargeattn(sparse_input, kwargs)
        else:
            raise ValueError(f"Unsupported sparse attention method: {self.config.method}")

        self.stats.add_official_kernel(skipped_fraction)
        return _restore_output(out, sparse_input)

    def _should_try_official_kernel(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kwargs: dict[str, Any],
        transformer_options: dict[str, Any],
    ) -> bool:
        if kwargs.get("mask") is not None:
            return False
        if _attention_tokens(q, kwargs) != _attention_tokens(k, kwargs):
            return False
        if _attention_tokens(k, kwargs) != _attention_tokens(v, kwargs):
            return False
        tokens = _attention_tokens(q, kwargs)
        if tokens < self.config.min_tokens or tokens > self.config.max_tokens:
            return False
        block_type = transformer_options.get("block_type", "unknown")
        if self.config.apply_to != "all" and block_type != self.config.apply_to:
            return False
        return True

    def _mark_denoise_step(self, transformer_options: dict[str, Any]) -> None:
        if transformer_options.get("block_type") == "double" and int(transformer_options.get("block_index", -1)) == 0:
            self.denoise_step += 1

    def _run_pisa(self, x: SparseInput, kwargs: dict[str, Any]) -> tuple[torch.Tensor, float]:
        piecewise_sparse_attention = self._get_pisa_kernel()
        _ensure_pisa_triton_cuda_allocator(x.q.device)
        out = piecewise_sparse_attention(
            x.q,
            x.k,
            x.v,
            density=self.config.density,
            block_size=self.config.block_size,
            scale=kwargs.get("scale"),
        )
        return out.to(dtype=x.q.dtype), _pisa_estimated_skipped_fraction(x.tokens, self.config.block_size, self.config.density)

    def _get_pisa_kernel(self):
        if self._pisa_kernel is not None:
            return self._pisa_kernel
        try:
            from piecewise_attn import piecewise_sparse_attention
        except ImportError as exc:
            raise RuntimeError(
                "PISA official sparse attention requires the official "
                "`piecewise_attn` package. Install xie-lab-ml/piecewise-sparse-attention "
                "in this conda env before running the PISA Flux canvas."
            ) from exc
        self._pisa_kernel = piecewise_sparse_attention
        return piecewise_sparse_attention

    def precompile_pisa(self) -> bool:
        if self.config.method != "pisa" or self.config.precompile_tokens <= 0:
            return False
        if not torch.cuda.is_available():
            return False

        device = torch.device("cuda", torch.cuda.current_device())
        tokens = self.config.precompile_tokens
        heads = self.config.precompile_heads
        dim = self.config.precompile_head_dim
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        key = (device.index, tokens, heads, dim, dtype, self.config.block_size, self.config.density)
        if key in _PISA_PRECOMPILED_CONFIGS:
            return True
        piecewise_sparse_attention = self._get_pisa_kernel()
        _ensure_pisa_triton_cuda_allocator(device)
        with torch.inference_mode():
            q = torch.randn((1, heads, tokens, dim), device=device, dtype=dtype)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            out = piecewise_sparse_attention(
                q,
                k,
                v,
                density=self.config.density,
                block_size=self.config.block_size,
                scale=dim ** -0.5,
            )
            torch.cuda.synchronize(device)
            del q, k, v, out
            torch.cuda.empty_cache()
        _PISA_PRECOMPILED_CONFIGS.add(key)
        return True

    def _run_spargeattn(self, x: SparseInput, kwargs: dict[str, Any]) -> tuple[torch.Tensor, float | None]:
        try:
            from spas_sage_attn import spas_sage2_attn_meansim_topk_cuda
        except ImportError as exc:
            raise RuntimeError(
                "SpargeAttn official sparse attention requires the official "
                "`spas_sage_attn` package. Build/install THU-ML/SpargeAttn in this "
                "conda env before running the SpargeAttn Flux canvas."
            ) from exc

        _ensure_triton_cuda_allocator(x.q.device)
        result = spas_sage2_attn_meansim_topk_cuda(
            x.q,
            x.k,
            x.v,
            is_causal=False,
            scale=kwargs.get("scale"),
            topk=self.config.topk,
            tensor_layout="HND",
            return_sparsity=True,
        )
        if isinstance(result, tuple):
            out, skipped_fraction = result
        else:
            out, skipped_fraction = result, None
        return out.to(dtype=x.q.dtype), skipped_fraction


def _call_attention(func: AttentionFn, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, kwargs: dict[str, Any]) -> torch.Tensor:
    clean_kwargs = dict(kwargs)
    clean_kwargs.pop("_inside_attn_wrapper", None)
    return func(q, k, v, heads, **clean_kwargs)


def _attention_tokens(x: torch.Tensor, kwargs: dict[str, Any]) -> int:
    if kwargs.get("skip_reshape", False):
        return int(x.shape[-2])
    return int(x.shape[1])


def _canonicalize(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, kwargs: dict[str, Any]) -> SparseInput | None:
    skip_reshape = bool(kwargs.get("skip_reshape", False))
    skip_output_reshape = bool(kwargs.get("skip_output_reshape", False))
    enable_gqa = bool(kwargs.get("enable_gqa", False))
    if skip_reshape:
        if q.ndim != 4:
            return None
        batch, q_heads, tokens, dim = q.shape
        if q_heads != heads:
            return None
        if enable_gqa and q.shape[-3] != k.shape[-3]:
            if q.shape[-3] % k.shape[-3] != 0:
                return None
            repeats = q.shape[-3] // k.shape[-3]
            k = k.repeat_interleave(repeats, dim=-3)
            v = v.repeat_interleave(repeats, dim=-3)
        if k.shape[:3] != q.shape[:3] or v.shape[:3] != q.shape[:3]:
            return None
        return SparseInput(q.contiguous(), k.contiguous(), v.contiguous(), batch, heads, tokens, dim, skip_output_reshape)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or q.shape[-1] % heads != 0:
        return None
    if q.shape[0] != k.shape[0] or q.shape[1] != k.shape[1] or q.shape[0] != v.shape[0] or q.shape[1] != v.shape[1]:
        return None
    batch, tokens, inner = q.shape
    dim = inner // heads
    if k.shape[-1] != inner or v.shape[-1] != inner:
        return None
    qh = q.view(batch, tokens, heads, dim).transpose(1, 2).contiguous()
    kh = k.view(batch, tokens, heads, dim).transpose(1, 2).contiguous()
    vh = v.view(batch, tokens, heads, dim).transpose(1, 2).contiguous()
    return SparseInput(qh, kh, vh, batch, heads, tokens, dim, skip_output_reshape)


def _restore_output(out: torch.Tensor, x: SparseInput) -> torch.Tensor:
    if x.skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(x.batch, x.tokens, x.heads * x.dim)


def _pisa_estimated_skipped_fraction(tokens: int, block_size: int, density: float) -> float:
    num_blocks = max(1, math.ceil(tokens / max(block_size, 1)))
    selected_blocks = max(1, int(float(density) * num_blocks))
    return 1.0 - (selected_blocks / num_blocks)


def _ensure_pisa_triton_cuda_allocator(device: torch.device) -> None:
    global _PISA_TRITON_ALLOCATOR_DEVICE
    device = torch.device(device)
    if _PISA_TRITON_ALLOCATOR_DEVICE == device:
        return
    _ensure_triton_cuda_allocator(device)
    _PISA_TRITON_ALLOCATOR_DEVICE = device


def _ensure_triton_cuda_allocator(device: torch.device) -> None:
    try:
        import triton
    except ImportError:
        return

    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device=device, dtype=torch.int8))


def sparse_attention_backend_spec(method: str, **kwargs) -> dict[str, Any]:
    if method == "pisa":
        return {
            "backend": "official_gpu_kernel",
            "method": method,
            "package": "piecewise_attn",
            "entrypoint": "piecewise_attn.piecewise_sparse_attention",
            "source": "https://github.com/xie-lab-ml/piecewise-sparse-attention",
            "inputs": kwargs,
        }
    if method == "spargeattn":
        return {
            "backend": "official_gpu_kernel",
            "method": method,
            "package": "spas_sage_attn",
            "entrypoint": "spas_sage_attn.spas_sage2_attn_meansim_topk_cuda",
            "source": "https://github.com/thu-ml/SpargeAttn",
            "inputs": kwargs,
        }
    raise ValueError(f"Unsupported official sparse attention method: {method}")


def make_sparse_attention_config(method: str, **kwargs) -> SparseAttentionConfig:
    return SparseAttentionConfig(method=method, **kwargs)
