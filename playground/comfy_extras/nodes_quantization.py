from __future__ import annotations

import importlib
import logging
import builtins
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from comfy_api.latest import ComfyExtension, io

import comfy.patcher_extension


_QUANTIZED_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}
if not hasattr(builtins, "_AUTO_DEPLOY_SELECTIVE_QUANTIZATION_STATES"):
    builtins._AUTO_DEPLOY_SELECTIVE_QUANTIZATION_STATES = []
_SELECTIVE_QUANTIZATION_STATES: list[Any] = builtins._AUTO_DEPLOY_SELECTIVE_QUANTIZATION_STATES
_TORCHAO_IMPORT_PATCHED = False


def _patch_duplicate_fake_registration() -> None:
    """Work around nightly MSLK registering fake/meta kernels twice.

    The CUDA kernels remain unchanged; this only ignores duplicate fake-kernel
    registration during TorchAO import when MSLK already registered Meta.
    """
    global _TORCHAO_IMPORT_PATCHED
    if _TORCHAO_IMPORT_PATCHED:
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
    _TORCHAO_IMPORT_PATCHED = True


def _parse_skip_modules(skip_modules: str) -> tuple[str, ...]:
    raw = skip_modules.replace("\n", ",").split(",")
    return tuple(item.strip() for item in raw if item.strip())


def _import_attr(candidates: tuple[tuple[str, str], ...]):
    _patch_duplicate_fake_registration()
    errors = []
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        except Exception as exc:  # pragma: no cover - depends on optional packages
            errors.append(f"{module_name}.{attr_name}: {exc}")
    raise RuntimeError("Could not import required TorchAO symbol:\n" + "\n".join(errors))


def _make_torchao_config(recipe: str):
    if recipe == "float8_dynamic":
        config_cls = _import_attr((
            ("torchao.quantization", "Float8DynamicActivationFloat8WeightConfig"),
        ))
        return config_cls()
    if recipe == "mxfp8_dynamic":
        config_cls = _import_attr((
            ("torchao.prototype.mx_formats.inference_workflow", "MXDynamicActivationMXWeightConfig"),
            ("torchao.quantization", "MXDynamicActivationMXWeightConfig"),
        ))
        return config_cls(
            activation_dtype=torch.float8_e4m3fn,
            weight_dtype=torch.float8_e4m3fn,
        )
    if recipe == "nvfp4_dynamic":
        config_cls = _import_attr((
            ("torchao.prototype.mx_formats.inference_workflow", "NVFP4DynamicActivationNVFP4WeightConfig"),
            ("torchao.quantization", "NVFP4DynamicActivationNVFP4WeightConfig"),
        ))
        return config_cls(use_dynamic_per_tensor_scale=True, use_triton_kernel=True)
    raise ValueError(f"Unknown TorchAO quantization recipe: {recipe}")


def _fresh_model_patcher(model):
    cached_init = getattr(model, "cached_patcher_init", None)
    if cached_init is not None:
        init_fn, init_args = cached_init
        try:
            fresh = init_fn(*init_args, disable_dynamic=True)
        except TypeError:
            fresh = init_fn(*init_args)
        fresh.model_options = deepcopy(getattr(model, "model_options", {"transformer_options": {}}))
        return fresh
    logging.warning("TorchAOQuantizeModel had to quantize a cloned patcher; prefer models loaded from split loaders.")
    return model.clone(disable_dynamic=True)


def _diffusion_target(model_patcher):
    base_model = model_patcher.model
    return getattr(base_model, "diffusion_model", base_model)


def _make_filter_fn(skip_patterns: tuple[str, ...]):
    def filter_fn(first, second=None):
        if isinstance(first, str):
            name = first
            module = second
        else:
            module = first
            name = second or ""
        if not isinstance(module, torch.nn.Linear):
            return False
        return not any(pattern in str(name) for pattern in skip_patterns)

    return filter_fn


def torchao_prepare_sampling_wrapper(
    executor,
    model,
    noise_shape,
    conds,
    model_options=None,
    force_full_load=False,
    force_offload=False,
):
    transformer_options = model.model_options.get("transformer_options", {})
    quantization = transformer_options.get("quantization", {})
    if quantization.get("backend") in {"torchao", "torchao_selective"}:
        model.model.device = getattr(model, "load_device", model.model.device)
        return model.model, conds, []
    return executor(model, noise_shape, conds, model_options=model_options, force_full_load=force_full_load, force_offload=force_offload)


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_linear_ms(linear: torch.nn.Module, x: torch.Tensor, warmup_runs: int, benchmark_runs: int, benchmark_loops: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup_runs):
            linear(x)
        _sync_cuda()
        elapsed = 0.0
        for _ in range(benchmark_runs):
            if x.is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(benchmark_loops):
                    linear(x)
                end.record()
                torch.cuda.synchronize()
                elapsed += start.elapsed_time(end)
            else:
                import time

                start_time = time.perf_counter()
                for _ in range(benchmark_loops):
                    linear(x)
                elapsed += (time.perf_counter() - start_time) * 1000.0
        return elapsed / max(benchmark_runs * benchmark_loops, 1)


@dataclass
class SelectiveLinearQuantState:
    recipe: str
    min_speedup: float
    warmup_runs: int
    benchmark_runs: int
    benchmark_loops: int
    verbose: bool = False
    shape_results: dict[tuple[Any, ...], dict[str, Any]] = field(default_factory=dict)
    module_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        quantized = sum(1 for item in self.module_results.values() if item.get("quantized"))
        kept = sum(1 for item in self.module_results.values() if not item.get("quantized"))
        return {
            "recipe": self.recipe,
            "min_speedup": self.min_speedup,
            "benchmarked_linear_shapes": len(self.shape_results),
            "quantized_linear_modules": quantized,
            "kept_high_precision_linear_modules": kept,
        }


def _linear_shape_key(recipe: str, linear: torch.nn.Linear, x: torch.Tensor) -> tuple[Any, ...]:
    in_features = int(linear.in_features)
    out_features = int(linear.out_features)
    token_rows = int(x.numel() // in_features) if in_features else 0
    device_name = x.device.type
    if x.is_cuda:
        device_name = f"cuda:{torch.cuda.get_device_capability(x.device)}"
    return (
        recipe,
        token_rows,
        in_features,
        out_features,
        linear.bias is not None,
        str(x.dtype),
        device_name,
    )


def _benchmark_quant_decision(linear: torch.nn.Linear, x: torch.Tensor, state: SelectiveLinearQuantState) -> dict[str, Any]:
    if not x.is_cuda:
        return {"quantized": False, "reason": "non_cuda_input"}
    if x.shape[-1] != linear.in_features:
        return {"quantized": False, "reason": "input_feature_mismatch"}

    bench_x = x.detach()
    bf16_ms = _time_linear_ms(linear, bench_x, state.warmup_runs, state.benchmark_runs, state.benchmark_loops)
    try:
        quant_linear = torch.nn.Linear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device=bench_x.device,
            dtype=linear.weight.dtype,
        ).eval()
        quantize_ = _import_attr((("torchao.quantization", "quantize_"),))
        quantize_(quant_linear, _make_torchao_config(state.recipe))
        quant_ms = _time_linear_ms(quant_linear, bench_x, state.warmup_runs, state.benchmark_runs, state.benchmark_loops)
    except Exception as exc:
        return {"quantized": False, "bf16_ms": bf16_ms, "reason": type(exc).__name__, "error": str(exc).splitlines()[0]}
    finally:
        locals().pop("quant_linear", None)

    speedup = bf16_ms / quant_ms if quant_ms > 0 else 0.0
    return {
        "quantized": speedup >= state.min_speedup,
        "bf16_ms": bf16_ms,
        "quant_ms": quant_ms,
        "speedup": speedup,
    }


class SelectiveTorchAOLinear(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, module_name: str, state: SelectiveLinearQuantState):
        super().__init__()
        self.linear = linear
        self.module_name = module_name
        self.state = state
        self.decided = False

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    def _decide(self, x: torch.Tensor) -> None:
        if self.decided:
            return
        key = _linear_shape_key(self.state.recipe, self.linear, x)
        result = self.state.shape_results.get(key)
        if result is None:
            result = _benchmark_quant_decision(self.linear, x, self.state)
            self.state.shape_results[key] = result
        module_result = dict(result)
        module_result["shape_key"] = list(key)
        if result.get("quantized"):
            try:
                quantize_ = _import_attr((("torchao.quantization", "quantize_"),))
                quantize_(self.linear, _make_torchao_config(self.state.recipe))
            except Exception as exc:
                module_result.update({"quantized": False, "reason": type(exc).__name__, "error": str(exc).splitlines()[0]})
        self.state.module_results[self.module_name] = module_result
        if self.state.verbose:
            logging.info("SelectiveTorchAOLinear %s decision: %s", self.module_name, module_result)
        self.decided = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._decide(x)
        return self.linear(x)


def _wrap_selective_linear_modules(module: torch.nn.Module, state: SelectiveLinearQuantState, skip_patterns: tuple[str, ...], prefix: str = "") -> int:
    count = 0
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, SelectiveTorchAOLinear):
            continue
        if isinstance(child, torch.nn.Linear):
            if any(pattern in full_name for pattern in skip_patterns):
                continue
            setattr(module, child_name, SelectiveTorchAOLinear(child, full_name, state))
            count += 1
            continue
        count += _wrap_selective_linear_modules(child, state, skip_patterns, full_name)
    return count


def _selective_quantize_with_torchao(
    model,
    recipe: str,
    min_speedup: float,
    skip_modules: str,
    warmup_runs: int,
    benchmark_runs: int,
    benchmark_loops: int,
    cache_quantized_model: bool,
    verbose: bool,
):
    try:
        _import_attr((("torchao.quantization", "quantize_"),))
    except RuntimeError as exc:
        raise RuntimeError(
            "SelectiveTorchAOQuantizeModel requires torchao in the active environment. "
            "Install the CUDA 13.0/nightly TorchAO build that matches this repo's PyTorch build."
        ) from exc

    cache_key = (id(model), "selective", recipe, min_speedup, skip_modules, warmup_runs, benchmark_runs, benchmark_loops)
    if cache_quantized_model and cache_key in _QUANTIZED_MODEL_CACHE:
        return _QUANTIZED_MODEL_CACHE[cache_key]

    quantized_model = _fresh_model_patcher(model)
    target_device = getattr(quantized_model, "load_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    quantized_model.model.to(target_device)
    quantized_model.model.device = target_device
    state = SelectiveLinearQuantState(
        recipe=recipe,
        min_speedup=min_speedup,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        benchmark_loops=benchmark_loops,
        verbose=verbose,
    )
    _SELECTIVE_QUANTIZATION_STATES.append(state)
    target = _diffusion_target(quantized_model)
    wrapped_count = _wrap_selective_linear_modules(target, state, _parse_skip_modules(skip_modules))
    quantized_model.model_options.setdefault("transformer_options", {})
    quantized_model.model_options["transformer_options"]["quantization"] = {
        "backend": "torchao_selective",
        "recipe": recipe,
        "skip_modules": skip_modules,
        "wrapped_linear_modules": wrapped_count,
    }
    quantized_model.model_options["transformer_options"]["selective_quantization_state"] = state
    quantized_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
        f"selective_torchao_quantization_{recipe}",
        torchao_prepare_sampling_wrapper,
    )
    if cache_quantized_model:
        _QUANTIZED_MODEL_CACHE[cache_key] = quantized_model
    return quantized_model


def selective_quantization_summaries() -> list[dict[str, Any]]:
    summaries = [state.summary() for state in _SELECTIVE_QUANTIZATION_STATES]
    for model in _QUANTIZED_MODEL_CACHE.values():
        transformer_options = model.model_options.get("transformer_options", {})
        state = transformer_options.get("selective_quantization_state")
        if state is not None:
            summaries.append(state.summary())
    return summaries


def _quantize_with_torchao(model, recipe: str, skip_modules: str, cache_quantized_model: bool):
    try:
        quantize_ = _import_attr((("torchao.quantization", "quantize_"),))
    except RuntimeError as exc:
        raise RuntimeError(
            "TorchAOQuantizeModel requires torchao in the active environment. "
            "Install the CUDA 13.0/nightly TorchAO build that matches this repo's PyTorch build."
        ) from exc

    cache_key = (id(model), recipe, skip_modules)
    if cache_quantized_model and cache_key in _QUANTIZED_MODEL_CACHE:
        return _QUANTIZED_MODEL_CACHE[cache_key]

    if hasattr(torch, "__future__") and hasattr(torch.__future__, "set_overwrite_module_params_on_conversion"):
        torch.__future__.set_overwrite_module_params_on_conversion(True)
    quantized_model = _fresh_model_patcher(model)
    target_device = getattr(quantized_model, "load_device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    quantized_model.model.to(target_device)
    quantized_model.model.device = target_device
    target = _diffusion_target(quantized_model)
    config = _make_torchao_config(recipe)
    filter_fn = _make_filter_fn(_parse_skip_modules(skip_modules))
    quantize_(target, config, filter_fn=filter_fn)
    quantized_model.model_options.setdefault("transformer_options", {})
    quantized_model.model_options["transformer_options"]["quantization"] = {
        "backend": "torchao",
        "recipe": recipe,
        "skip_modules": skip_modules,
    }
    quantized_model.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
        f"torchao_quantization_{recipe}",
        torchao_prepare_sampling_wrapper,
    )
    if cache_quantized_model:
        _QUANTIZED_MODEL_CACHE[cache_key] = quantized_model
    return quantized_model


class TorchAOQuantizeModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TorchAOQuantizeModel",
            display_name="TorchAO Quantize Model",
            category="advanced/model",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "recipe",
                    options=["float8_dynamic", "mxfp8_dynamic", "nvfp4_dynamic"],
                    default="float8_dynamic",
                    advanced=True,
                ),
                io.String.Input(
                    "skip_modules",
                    default="img_in,txt_in,time_in,vector_in,guidance_in,final_layer",
                    multiline=False,
                    advanced=True,
                    tooltip="Comma-separated module-name substrings to keep out of TorchAO quantization.",
                ),
                io.Boolean.Input("cache_quantized_model", default=True, advanced=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, recipe, skip_modules, cache_quantized_model) -> io.NodeOutput:
        return io.NodeOutput(_quantize_with_torchao(model, recipe, skip_modules, cache_quantized_model))


class SelectiveTorchAOQuantizeModel(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SelectiveTorchAOQuantizeModel",
            display_name="Selective TorchAO Quantize Model",
            category="advanced/model",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "recipe",
                    options=["float8_dynamic", "mxfp8_dynamic", "nvfp4_dynamic"],
                    default="nvfp4_dynamic",
                    advanced=True,
                ),
                io.Float.Input("min_speedup", default=1.05, min=1.0, max=10.0, step=0.01, advanced=True),
                io.String.Input(
                    "skip_modules",
                    default="img_in,txt_in,time_in,vector_in,guidance_in,final_layer",
                    multiline=False,
                    advanced=True,
                    tooltip="Comma-separated module-name substrings to keep high precision.",
                ),
                io.Int.Input("warmup_runs", default=3, min=0, max=100, step=1, advanced=True),
                io.Int.Input("benchmark_runs", default=8, min=1, max=100, step=1, advanced=True),
                io.Int.Input("benchmark_loops", default=4, min=1, max=100, step=1, advanced=True),
                io.Boolean.Input("cache_quantized_model", default=True, advanced=True),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        recipe,
        min_speedup,
        skip_modules,
        warmup_runs,
        benchmark_runs,
        benchmark_loops,
        cache_quantized_model,
        verbose,
    ) -> io.NodeOutput:
        return io.NodeOutput(_selective_quantize_with_torchao(
            model,
            recipe=recipe,
            min_speedup=min_speedup,
            skip_modules=skip_modules,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            benchmark_loops=benchmark_loops,
            cache_quantized_model=cache_quantized_model,
            verbose=verbose,
        ))


class QuantizationExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            TorchAOQuantizeModel,
            SelectiveTorchAOQuantizeModel,
        ]


def comfy_entrypoint():
    return QuantizationExtension()
