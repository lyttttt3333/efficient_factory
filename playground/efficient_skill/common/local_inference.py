from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import folder_paths
import nodes
import torch


LOCAL_EXTRA_NODE_FILES = (
    "nodes_custom_sampler.py",
    "nodes_flux.py",
    "nodes_easycache.py",
    "nodes_teacache.py",
    "nodes_reuse_cache.py",
    "nodes_quantization.py",
    "nodes_sparse_attention.py",
    "nodes_torch_compile.py",
)


_EXTRA_NODES_READY = False


@dataclass
class WorkflowExecution:
    save_results: dict[str, dict[str, Any]]
    node_wall_times_s: dict[str, float]
    denoise_wall_time_s: float = 0.0
    total_steps: int = 0
    skipped_steps: int = 0
    attention_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowRunResult:
    wall_time_s: float
    denoise_wall_time_s: float
    image_path: Path
    total_steps: int
    skipped_steps: int
    node_wall_times_s: dict[str, float] = field(default_factory=dict)
    attention_stats: dict[str, Any] = field(default_factory=dict)


async def _load_local_extra_nodes() -> None:
    await nodes.init_public_apis()
    extras_dir = Path(__file__).resolve().parents[2] / "comfy_extras"
    for node_file in LOCAL_EXTRA_NODE_FILES:
        node_path = extras_dir / node_file
        if not await nodes.load_custom_node(str(node_path), module_parent="comfy_extras"):
            raise RuntimeError(f"Failed to load local inference node: {node_path}")


def setup_local_runtime(output_root: Path) -> None:
    global _EXTRA_NODES_READY
    runtime_root = output_root.parent / "runtime"
    input_root = runtime_root / "input"
    temp_root = runtime_root / "temp"
    user_root = runtime_root / "user"
    for path in (output_root, input_root, temp_root, user_root):
        path.mkdir(parents=True, exist_ok=True)

    folder_paths.set_output_directory(str(output_root))
    folder_paths.set_input_directory(str(input_root))
    folder_paths.set_temp_directory(str(temp_root))
    folder_paths.set_user_directory(str(user_root))

    if not _EXTRA_NODES_READY:
        asyncio.run(_load_local_extra_nodes())
        _EXTRA_NODES_READY = True


def output_images_from_save_results(save_results: dict[str, dict[str, Any]], output_root: Path) -> list[Path]:
    images: list[Path] = []
    for node_output in save_results.values():
        for image in node_output.get("images", []):
            if image.get("type", "output") != "output":
                continue
            subfolder = image.get("subfolder") or ""
            images.append(output_root / subfolder / image["filename"])
    return images


def _is_ref(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and isinstance(value[1], int)


def _cacheable_literal_inputs(inputs: dict[str, Any]) -> tuple[tuple[str, Any], ...] | None:
    literal_items = []
    for key, value in sorted(inputs.items()):
        if _is_ref(value):
            return None
        if isinstance(value, (str, int, float, bool, type(None))):
            literal_items.append((key, value))
            continue
        return None
    return tuple(literal_items)


def _normalize_output(raw_output: Any) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    if hasattr(raw_output, "result"):
        result = raw_output.result
        if result is None:
            result = ()
        return tuple(result), getattr(raw_output, "ui", None)
    if isinstance(raw_output, tuple):
        return raw_output, None
    if isinstance(raw_output, dict):
        result = raw_output.get("result", ())
        if result is None:
            result = ()
        return tuple(result), raw_output.get("ui")
    if raw_output is None:
        return (), None
    return (raw_output,), None


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _sampler_step_stats(inputs: dict[str, Any]) -> tuple[int, int]:
    sigmas = inputs.get("sigmas")
    default_total = max(int(sigmas.shape[-1]) - 1, 0) if sigmas is not None else 0
    guider = inputs.get("guider")
    model_options = getattr(guider, "model_options", {})
    transformer_options = model_options.get("transformer_options", {}) if isinstance(model_options, dict) else {}

    for key in ("easycache", "teacache", "reusecache"):
        holder = transformer_options.get(key)
        if holder is None:
            continue
        total = int(getattr(holder, "last_total_steps", 0) or default_total)
        skipped = int(getattr(holder, "last_steps_skipped", 0))
        return total, skipped
    return default_total, 0


def _sampler_attention_stats(inputs: dict[str, Any]) -> dict[str, Any]:
    guider = inputs.get("guider")
    model_options = getattr(guider, "model_options", {})
    transformer_options = model_options.get("transformer_options", {}) if isinstance(model_options, dict) else {}
    holder = transformer_options.get("sparse_attention")
    if holder is not None and hasattr(holder, "snapshot"):
        return holder.snapshot()
    return {}


class LocalWorkflowRunner:
    def __init__(self) -> None:
        self.literal_cache: dict[tuple[str, tuple[tuple[str, Any], ...]], tuple[Any, ...]] = {}

    def _resolve_inputs(self, inputs: dict[str, Any], outputs: dict[str, tuple[Any, ...]], evaluate_node) -> dict[str, Any]:
        resolved = {}
        for key, value in inputs.items():
            if _is_ref(value):
                evaluate_node(value[0])
                resolved[key] = outputs[value[0]][value[1]]
            else:
                resolved[key] = value
        return resolved

    def _execute_node(self, class_type: str, inputs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
        cache_key = None
        literal_inputs = _cacheable_literal_inputs(inputs)
        if literal_inputs is not None and class_type in {"UNETLoader", "DualCLIPLoader", "VAELoader", "KSamplerSelect"}:
            cache_key = (class_type, literal_inputs)
            cached = self.literal_cache.get(cache_key)
            if cached is not None:
                return cached, None

        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        function_name = getattr(class_def, "FUNCTION", None)
        if function_name is None:
            raw_output = class_def.execute(**inputs)
        else:
            instance = class_def()
            raw_output = getattr(instance, function_name)(**inputs)

        outputs, ui = _normalize_output(raw_output)
        if cache_key is not None:
            self.literal_cache[cache_key] = outputs
        return outputs, ui

    def run(self, workflow: dict[str, dict[str, Any]]) -> WorkflowExecution:
        outputs: dict[str, tuple[Any, ...]] = {}
        save_results: dict[str, dict[str, Any]] = {}
        node_wall_times_s: dict[str, float] = {}
        denoise_wall_time_s = 0.0
        total_steps = 0
        skipped_steps = 0
        attention_stats: dict[str, Any] = {}
        visiting: set[str] = set()

        def evaluate_node(node_id: str) -> None:
            nonlocal denoise_wall_time_s, total_steps, skipped_steps, attention_stats
            if node_id in outputs:
                return
            if node_id in visiting:
                raise RuntimeError(f"Cycle detected while evaluating node {node_id}")
            visiting.add(node_id)
            node = workflow[node_id]
            inputs = self._resolve_inputs(node.get("inputs", {}), outputs, evaluate_node)
            class_type = node["class_type"]
            if class_type == "SamplerCustomAdvanced":
                _sync_cuda()
            start = time.perf_counter()
            node_outputs, save_result = self._execute_node(node["class_type"], inputs)
            if class_type == "SamplerCustomAdvanced":
                _sync_cuda()
            elapsed = time.perf_counter() - start
            node_wall_times_s[node_id] = elapsed
            outputs[node_id] = node_outputs
            if save_result is not None:
                save_results[node_id] = save_result
            if class_type == "SamplerCustomAdvanced":
                steps, skipped = _sampler_step_stats(inputs)
                denoise_wall_time_s += elapsed
                total_steps += steps
                skipped_steps += skipped
                attention_stats = _sampler_attention_stats(inputs)
            visiting.remove(node_id)

        for node_id in sorted(workflow, key=lambda value: int(value)):
            evaluate_node(node_id)
        return WorkflowExecution(
            save_results=save_results,
            node_wall_times_s=node_wall_times_s,
            denoise_wall_time_s=denoise_wall_time_s,
            total_steps=total_steps,
            skipped_steps=skipped_steps,
            attention_stats=attention_stats,
        )


def make_runner() -> LocalWorkflowRunner:
    return LocalWorkflowRunner()


def run_workflow(runner: LocalWorkflowRunner, workflow: dict[str, dict[str, Any]], output_root: Path) -> WorkflowRunResult:
    _sync_cuda()
    start = time.perf_counter()
    with torch.inference_mode():
        execution = runner.run(workflow)
    _sync_cuda()
    elapsed = time.perf_counter() - start
    images = output_images_from_save_results(execution.save_results, output_root)
    if not images:
        raise RuntimeError("No output image found")
    return WorkflowRunResult(
        wall_time_s=elapsed,
        denoise_wall_time_s=execution.denoise_wall_time_s,
        image_path=images[-1],
        total_steps=execution.total_steps,
        skipped_steps=execution.skipped_steps,
        node_wall_times_s=execution.node_wall_times_s,
        attention_stats=execution.attention_stats,
    )
