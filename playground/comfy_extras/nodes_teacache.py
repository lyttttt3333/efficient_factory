from __future__ import annotations

import logging

import torch
from comfy_api.latest import ComfyExtension, io

import comfy.model_patcher


FLUX_COEFFICIENTS = (
    4.98651651e2,
    -2.83781631e2,
    5.58554382e1,
    -3.82021401,
    2.64230861e-1,
)


def _poly1d(coefficients: tuple[float, ...], x: float) -> float:
    result = 0.0
    degree = len(coefficients) - 1
    for index, coeff in enumerate(coefficients):
        result += coeff * (x ** (degree - index))
    return result


class TeaCacheHolder:
    def __init__(
        self,
        rel_l1_thresh: float,
        start_percent: float,
        end_percent: float,
        max_skip_steps: int,
        cache_device: str,
        verbose: bool = False,
    ):
        self.name = "TeaCache"
        self.rel_l1_thresh = rel_l1_thresh
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.max_skip_steps = max_skip_steps
        self.cache_device = cache_device
        self.verbose = verbose
        self.start_t = 0.0
        self.end_t = 0.0
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = 0.0
        self.skipped_in_a_row = 0
        self.total_steps_seen = 0
        self.total_steps_skipped = 0
        self.last_total_steps = 0
        self.last_steps_skipped = 0

    def clone(self):
        return TeaCacheHolder(
            rel_l1_thresh=self.rel_l1_thresh,
            start_percent=self.start_percent,
            end_percent=self.end_percent,
            max_skip_steps=self.max_skip_steps,
            cache_device=self.cache_device,
            verbose=self.verbose,
        )

    def prepare_timesteps(self, model_sampling):
        self.start_t = model_sampling.percent_to_sigma(self.start_percent)
        self.end_t = model_sampling.percent_to_sigma(self.end_percent)
        return self

    def has_residual(self) -> bool:
        return self.previous_residual is not None

    def should_calculate(self, modulated_input: torch.Tensor, timestep: torch.Tensor) -> bool:
        self.total_steps_seen += 1
        if not self._is_in_cache_window(timestep):
            self.previous_modulated_input = self._store_tensor(modulated_input)
            self.accumulated_rel_l1_distance = 0.0
            self.skipped_in_a_row = 0
            return True

        current = self._store_tensor(modulated_input)
        if self.previous_modulated_input is None or self.previous_residual is None:
            self.previous_modulated_input = current
            self.accumulated_rel_l1_distance = 0.0
            self.skipped_in_a_row = 0
            return True

        previous = self.previous_modulated_input
        rel_l1 = (current.float() - previous.float()).abs().mean() / previous.float().abs().mean().clamp_min(1e-6)
        self.accumulated_rel_l1_distance += abs(_poly1d(FLUX_COEFFICIENTS, float(rel_l1.item())))
        self.previous_modulated_input = current

        if self.accumulated_rel_l1_distance < self.rel_l1_thresh and self.skipped_in_a_row < self.max_skip_steps:
            self.skipped_in_a_row += 1
            self.total_steps_skipped += 1
            if self.verbose:
                logging.info(
                    "TeaCache - skipping step; accumulated_rel_l1_distance: %.6f, threshold: %.6f",
                    self.accumulated_rel_l1_distance,
                    self.rel_l1_thresh,
                )
            return False

        self.accumulated_rel_l1_distance = 0.0
        self.skipped_in_a_row = 0
        return True

    def update_residual(self, output: torch.Tensor, original: torch.Tensor) -> None:
        self.previous_residual = self._store_tensor(output.detach() - original.detach())

    def reset(self):
        self.previous_modulated_input = None
        self.previous_residual = None
        self.accumulated_rel_l1_distance = 0.0
        self.skipped_in_a_row = 0
        self.total_steps_seen = 0
        self.total_steps_skipped = 0
        return self

    def _is_in_cache_window(self, timestep: torch.Tensor) -> bool:
        if timestep is None:
            return True
        return (timestep[0] <= self.start_t).item() and (timestep[0] > self.end_t).item()

    def _store_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach()
        if self.cache_device == "cpu":
            return tensor.float().cpu()
        return tensor.clone()


def teacache_sample_wrapper(executor, *args, **kwargs):
    try:
        guider = executor.class_obj
        orig_model_options = guider.model_options
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        holder = guider.model_options["transformer_options"]["teacache"].clone().prepare_timesteps(guider.model_patcher.model.model_sampling)
        guider.model_options["transformer_options"]["teacache"] = holder
        logging.info(
            "TeaCache enabled - rel_l1_thresh: %.4f, start_percent: %.2f, end_percent: %.2f, max_skip_steps: %d",
            holder.rel_l1_thresh,
            holder.start_percent,
            holder.end_percent,
            holder.max_skip_steps,
        )
        return executor(*args, **kwargs)
    finally:
        holder = guider.model_options["transformer_options"]["teacache"]
        total_steps = holder.total_steps_seen
        skipped = holder.total_steps_skipped
        holder.last_total_steps = total_steps
        holder.last_steps_skipped = skipped
        original_holder = orig_model_options["transformer_options"].get("teacache")
        if original_holder is not None:
            original_holder.last_total_steps = total_steps
            original_holder.last_steps_skipped = skipped
        speedup = total_steps / (total_steps - skipped) if total_steps > skipped else 1.0
        logging.info("TeaCache - skipped %d/%d model calls (%.2fx theoretical DiT speedup).", skipped, total_steps, speedup)
        holder.reset()
        guider.model_options = orig_model_options


class TeaCacheNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TeaCache",
            display_name="TeaCache",
            description="Flux TeaCache-style residual cache.",
            category="advanced/model",
            is_experimental=True,
            inputs=[
                io.Model.Input("model", tooltip="The model to add TeaCache to."),
                io.Float.Input("rel_l1_thresh", min=0.0, default=0.4, max=1000.0, step=0.01, advanced=True),
                io.Float.Input("start_percent", min=0.0, default=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("end_percent", min=0.0, default=1.0, max=1.0, step=0.01, advanced=True),
                io.Int.Input("max_skip_steps", min=1, default=2, max=100, step=1, advanced=True),
                io.Combo.Input("cache_device", options=["default", "cpu"], advanced=True),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[
                io.Model.Output(tooltip="The model with TeaCache."),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: io.Model.Type,
        rel_l1_thresh: float,
        start_percent: float,
        end_percent: float,
        max_skip_steps: int,
        cache_device: str,
        verbose: bool,
    ) -> io.NodeOutput:
        model = model.clone()
        model.model_options["transformer_options"]["teacache"] = TeaCacheHolder(
            rel_l1_thresh=rel_l1_thresh,
            start_percent=start_percent,
            end_percent=end_percent,
            max_skip_steps=max_skip_steps,
            cache_device=cache_device,
            verbose=verbose,
        )
        model.add_wrapper_with_key("outer_sample", "teacache", teacache_sample_wrapper)
        return io.NodeOutput(model)


class TeaCacheExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [TeaCacheNode]


def comfy_entrypoint():
    return TeaCacheExtension()
