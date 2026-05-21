from __future__ import annotations

import logging

import comfy.model_patcher
import comfy.patcher_extension
import torch
from comfy_api.latest import ComfyExtension, io


class ReuseCacheHolder:
    def __init__(
        self,
        name: str,
        mode: str,
        start_percent: float,
        end_percent: float,
        warmup_steps: int,
        max_skip_steps: int,
        interval: int = 2,
        similarity_threshold: float = 0.05,
        subsample_factor: int = 8,
        delta_scale: float = 1.0,
        ema_decay: float = 0.5,
        verbose: bool = False,
    ) -> None:
        self.name = name
        self.mode = mode
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.warmup_steps = warmup_steps
        self.max_skip_steps = max_skip_steps
        self.interval = interval
        self.similarity_threshold = similarity_threshold
        self.subsample_factor = subsample_factor
        self.delta_scale = delta_scale
        self.ema_decay = ema_decay
        self.verbose = verbose
        self.start_t = 0.0
        self.end_t = 0.0
        self.steps_seen = 0
        self.skipped_in_a_row = 0
        self.total_steps_skipped = 0
        self.last_total_steps = 0
        self.last_steps_skipped = 0
        self.last_x = None
        self.last_output = None
        self.previous_output = None
        self.ema_output = None
        self.state_metadata = None

    def clone(self):
        return ReuseCacheHolder(
            name=self.name,
            mode=self.mode,
            start_percent=self.start_percent,
            end_percent=self.end_percent,
            warmup_steps=self.warmup_steps,
            max_skip_steps=self.max_skip_steps,
            interval=self.interval,
            similarity_threshold=self.similarity_threshold,
            subsample_factor=self.subsample_factor,
            delta_scale=self.delta_scale,
            ema_decay=self.ema_decay,
            verbose=self.verbose,
        )

    def prepare_timesteps(self, model_sampling):
        self.start_t = model_sampling.percent_to_sigma(self.start_percent)
        self.end_t = model_sampling.percent_to_sigma(self.end_percent)
        return self

    def reset(self):
        self.steps_seen = 0
        self.skipped_in_a_row = 0
        self.total_steps_skipped = 0
        self.last_x = None
        self.last_output = None
        self.previous_output = None
        self.ema_output = None
        self.state_metadata = None
        return self

    def in_cache_window(self, timestep: torch.Tensor) -> bool:
        return (timestep[0] <= self.start_t).item() and (timestep[0] > self.end_t).item()

    def check_metadata(self, x: torch.Tensor) -> bool:
        metadata = (x.device, x.dtype, x.shape)
        if self.state_metadata is None:
            self.state_metadata = metadata
            return True
        if metadata == self.state_metadata:
            return True
        logging.warning("%s - tensor metadata changed, resetting state", self.name)
        self.reset()
        self.state_metadata = metadata
        return False

    def subsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.subsample_factor > 1 and x.ndim >= 4:
            return x[..., ::self.subsample_factor, ::self.subsample_factor]
        return x

    def relative_input_change(self, x: torch.Tensor) -> float:
        if self.last_x is None:
            return float("inf")
        current = self.subsample(x.detach()).float()
        previous = self.subsample(self.last_x).float().to(current.device)
        return float((current - previous).abs().mean().div(previous.abs().mean().clamp_min(1e-6)).item())

    def schedule_skip(self) -> bool:
        if self.steps_seen <= self.warmup_steps:
            return False
        return (self.steps_seen - self.warmup_steps - 1) % self.interval == 0

    def should_skip(self, x: torch.Tensor, timestep: torch.Tensor) -> bool:
        self.steps_seen += 1
        if not self.in_cache_window(timestep):
            return False
        if self.last_output is None:
            return False
        if self.skipped_in_a_row >= self.max_skip_steps:
            return False
        if self.mode in {"periodic", "delta", "ema"}:
            return self.schedule_skip()
        if self.mode == "similarity":
            if self.steps_seen <= self.warmup_steps:
                return False
            rel_change = self.relative_input_change(x)
            if self.verbose:
                logging.info("%s - relative input change %.6f", self.name, rel_change)
            return rel_change < self.similarity_threshold
        raise ValueError(f"Unknown reuse cache mode: {self.mode}")

    def cached_output(self, x: torch.Tensor) -> torch.Tensor:
        self.total_steps_skipped += 1
        self.skipped_in_a_row += 1
        if self.mode == "delta" and self.previous_output is not None:
            delta = self.last_output - self.previous_output.to(self.last_output.device)
            return (self.last_output + self.delta_scale * delta).to(device=x.device, dtype=x.dtype)
        if self.mode == "ema" and self.ema_output is not None:
            return self.ema_output.to(device=x.device, dtype=x.dtype)
        return self.last_output.to(device=x.device, dtype=x.dtype)

    def update(self, x: torch.Tensor, output: torch.Tensor) -> None:
        output = output.detach()
        self.previous_output = self.last_output
        self.last_output = output.clone()
        if self.ema_output is None:
            self.ema_output = output.clone()
        else:
            self.ema_output = self.ema_decay * self.ema_output.to(output.device) + (1.0 - self.ema_decay) * output
            self.ema_output = self.ema_output.detach()
        self.last_x = x.detach().clone()
        self.skipped_in_a_row = 0


def reuse_predict_noise_wrapper(executor, *args, **kwargs):
    x: torch.Tensor = args[0]
    timestep: torch.Tensor = args[1]
    model_options: dict = args[2]
    holder: ReuseCacheHolder = model_options["transformer_options"]["reusecache"]
    holder.check_metadata(x)
    if holder.should_skip(x, timestep):
        if holder.verbose:
            logging.info("%s - skipping step %d", holder.name, holder.steps_seen)
        return holder.cached_output(x)
    output = executor(*args, **kwargs)
    holder.update(x, output)
    return output


def reuse_sample_wrapper(executor, *args, **kwargs):
    try:
        guider = executor.class_obj
        orig_model_options = guider.model_options
        guider.model_options = comfy.model_patcher.create_model_options_clone(orig_model_options)
        holder = guider.model_options["transformer_options"]["reusecache"].clone().prepare_timesteps(guider.model_patcher.model.model_sampling)
        guider.model_options["transformer_options"]["reusecache"] = holder
        logging.info(
            "%s enabled - mode: %s, interval: %d, warmup_steps: %d, max_skip_steps: %d",
            holder.name,
            holder.mode,
            holder.interval,
            holder.warmup_steps,
            holder.max_skip_steps,
        )
        return executor(*args, **kwargs)
    finally:
        holder = guider.model_options["transformer_options"]["reusecache"]
        total_steps = len(args[3]) - 1
        skipped = holder.total_steps_skipped
        holder.last_total_steps = total_steps
        holder.last_steps_skipped = skipped
        original_holder = orig_model_options["transformer_options"].get("reusecache")
        if original_holder is not None:
            original_holder.last_total_steps = total_steps
            original_holder.last_steps_skipped = skipped
        speedup = total_steps / (total_steps - skipped) if total_steps > skipped else 1.0
        logging.info("%s - skipped %d/%d model calls (%.2fx theoretical DiT speedup).", holder.name, skipped, total_steps, speedup)
        holder.reset()
        guider.model_options = orig_model_options


def _apply_reuse_cache(model, holder: ReuseCacheHolder):
    model = model.clone()
    model.model_options["transformer_options"]["reusecache"] = holder
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, holder.mode, reuse_sample_wrapper)
    model.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, holder.mode, reuse_predict_noise_wrapper)
    return io.NodeOutput(model)


class PeriodicReuseCacheNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PeriodicReuseCache",
            display_name="Periodic Reuse Cache",
            category="advanced/model",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("interval", default=2, min=1, max=100, step=1, advanced=True),
                io.Int.Input("warmup_steps", default=1, min=0, max=100, step=1, advanced=True),
                io.Int.Input("max_skip_steps", default=1, min=1, max=100, step=1, advanced=True),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, interval, warmup_steps, max_skip_steps, start_percent, end_percent, verbose) -> io.NodeOutput:
        return _apply_reuse_cache(model, ReuseCacheHolder("PeriodicReuseCache", "periodic", start_percent, end_percent, warmup_steps, max_skip_steps, interval=interval, verbose=verbose))


class SimilarityReuseCacheNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SimilarityReuseCache",
            display_name="Similarity Reuse Cache",
            category="advanced/model",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("similarity_threshold", default=1.00, min=0.0, max=100.0, step=0.01, advanced=True),
                io.Int.Input("subsample_factor", default=8, min=1, max=64, step=1, advanced=True),
                io.Int.Input("warmup_steps", default=1, min=0, max=100, step=1, advanced=True),
                io.Int.Input("max_skip_steps", default=1, min=1, max=100, step=1, advanced=True),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, similarity_threshold, subsample_factor, warmup_steps, max_skip_steps, start_percent, end_percent, verbose) -> io.NodeOutput:
        return _apply_reuse_cache(
            model,
            ReuseCacheHolder(
                "SimilarityReuseCache",
                "similarity",
                start_percent,
                end_percent,
                warmup_steps,
                max_skip_steps,
                similarity_threshold=similarity_threshold,
                subsample_factor=subsample_factor,
                verbose=verbose,
            ),
        )


class DeltaReuseCacheNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DeltaReuseCache",
            display_name="Delta Reuse Cache",
            category="advanced/model",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("delta_scale", default=1.0, min=-10.0, max=10.0, step=0.01, advanced=True),
                io.Int.Input("interval", default=2, min=1, max=100, step=1, advanced=True),
                io.Int.Input("warmup_steps", default=2, min=0, max=100, step=1, advanced=True),
                io.Int.Input("max_skip_steps", default=1, min=1, max=100, step=1, advanced=True),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, delta_scale, interval, warmup_steps, max_skip_steps, start_percent, end_percent, verbose) -> io.NodeOutput:
        return _apply_reuse_cache(model, ReuseCacheHolder("DeltaReuseCache", "delta", start_percent, end_percent, warmup_steps, max_skip_steps, interval=interval, delta_scale=delta_scale, verbose=verbose))


class EMAReuseCacheNode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EMAReuseCache",
            display_name="EMA Reuse Cache",
            category="advanced/model",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("ema_decay", default=0.5, min=0.0, max=0.999, step=0.01, advanced=True),
                io.Int.Input("interval", default=2, min=1, max=100, step=1, advanced=True),
                io.Int.Input("warmup_steps", default=1, min=0, max=100, step=1, advanced=True),
                io.Int.Input("max_skip_steps", default=1, min=1, max=100, step=1, advanced=True),
                io.Float.Input("start_percent", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Float.Input("end_percent", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True),
                io.Boolean.Input("verbose", default=False, advanced=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, ema_decay, interval, warmup_steps, max_skip_steps, start_percent, end_percent, verbose) -> io.NodeOutput:
        return _apply_reuse_cache(model, ReuseCacheHolder("EMAReuseCache", "ema", start_percent, end_percent, warmup_steps, max_skip_steps, interval=interval, ema_decay=ema_decay, verbose=verbose))


class ReuseCacheExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            PeriodicReuseCacheNode,
            SimilarityReuseCacheNode,
            DeltaReuseCacheNode,
            EMAReuseCacheNode,
        ]


def comfy_entrypoint():
    return ReuseCacheExtension()
