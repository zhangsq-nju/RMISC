from typing import cast

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback


def seed_worker(worker_id: int):
    import random

    import numpy as np

    seed = torch.initial_seed() % 2**32 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EvaluateAndSaveFinalStepCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_log = True
            control.should_evaluate = True
            control.should_save = True


class TimesFMTrainer(Trainer):
    """Trainer whose datasets already return complete batches."""

    def __init__(
        self,
        *args,
        point_loss: str = "mse",
        huber_beta: float = 1.0,
        quantile_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if point_loss not in {"mse", "huber"}:
            raise ValueError(f"Unsupported point_loss: {point_loss}")
        if huber_beta <= 0:
            raise ValueError("huber_beta must be > 0.")

        self.point_loss = point_loss
        self.huber_beta = float(huber_beta)
        self.quantile_loss_weight = float(quantile_loss_weight)
        self._loss_component_sums: dict[str, dict[str, torch.Tensor]] = {"train": {}, "eval": {}}
        self._loss_component_counts: dict[str, int] = {"train": 0, "eval": 0}

    @staticmethod
    def _unwrap_model(model):
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        return value.item() if isinstance(value, torch.Tensor) else float(value)

    def _record_loss_components(self, phase: str, **components):
        sums = self._loss_component_sums[phase]
        for name, value in components.items():
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(float(value), device=self.args.device)
            value = value.detach()
            if value.numel() != 1:
                value = value.mean()
            sums[name] = value if name not in sums else sums[name] + value.to(sums[name].device)
        self._loss_component_counts[phase] += 1

    def _flush_loss_components(self, logs: dict[str, float], phase: str, prefix: str = ""):
        count = self._loss_component_counts.get(phase, 0)
        if count <= 0:
            return

        for name, total in self._loss_component_sums[phase].items():
            logs[f"{prefix}{name}"] = self._to_float(total / count)
        self._loss_component_sums[phase] = {}
        self._loss_component_counts[phase] = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        future_values = inputs.pop("future_values")
        future_values_mask = inputs.pop("future_values_mask", None)
        raw_model = self._unwrap_model(model)
        phase = "train" if model.training else "eval"

        outputs = model(**inputs)
        target_len = future_values.shape[1]
        mean_predictions = outputs.mean_predictions[:, :target_len]

        if future_values_mask is None:
            future_values_mask = torch.isfinite(future_values).to(future_values.dtype)
        else:
            future_values_mask = future_values_mask.to(future_values.device, dtype=future_values.dtype)

        pred_finite = torch.isfinite(mean_predictions)
        valid_series = pred_finite.all(dim=1, keepdim=True)
        future_values_mask = future_values_mask * valid_series.to(future_values_mask.dtype)
        mean_predictions = torch.where(pred_finite, mean_predictions, torch.zeros_like(mean_predictions).detach())

        future_values = torch.nan_to_num(future_values, nan=0.0, posinf=0.0, neginf=0.0)
        denom = future_values_mask.sum().clamp_min(1.0)
        mse_loss = (F.mse_loss(mean_predictions, future_values, reduction="none") * future_values_mask).sum() / denom
        if self.point_loss == "huber":
            point_loss_values = F.smooth_l1_loss(
                mean_predictions,
                future_values,
                reduction="none",
                beta=self.huber_beta,
            )
            point_loss = (point_loss_values * future_values_mask).sum() / denom
        else:
            point_loss = mse_loss

        full_predictions = outputs.full_predictions[:, :target_len]
        full_predictions_finite = torch.isfinite(full_predictions)
        valid_series = valid_series * full_predictions_finite.flatten(start_dim=1).all(dim=1, keepdim=True)
        future_values_mask = future_values_mask * valid_series.to(future_values_mask.dtype)
        full_predictions = torch.where(
            full_predictions_finite,
            full_predictions,
            torch.zeros_like(full_predictions).detach(),
        )
        decode_index = min(raw_model.config.decode_index, full_predictions.shape[-1] - 1)
        quantile_indices = [i for i in range(full_predictions.shape[-1]) if i != decode_index]
        num_quantiles = min(len(quantile_indices), len(raw_model.config.quantiles))
        if num_quantiles > 0:
            quantile_indices = quantile_indices[:num_quantiles]
            quantile_levels = torch.tensor(raw_model.config.quantiles, device=full_predictions.device, dtype=full_predictions.dtype)
            quantile_levels = quantile_levels[:num_quantiles]
            quantile_preds = full_predictions[..., quantile_indices]
            errors = future_values.unsqueeze(-1) - quantile_preds
            quantile_losses = torch.maximum((quantile_levels - 1.0) * errors, quantile_levels * errors)
            quantile_mask = future_values_mask.unsqueeze(-1)
            quantile_loss = (quantile_losses * quantile_mask).sum() / (quantile_mask.sum().clamp_min(1.0) * num_quantiles)
        else:
            quantile_loss = point_loss.new_zeros(())

        loss = point_loss + self.quantile_loss_weight * quantile_loss
        valid_future_ratio = future_values_mask.sum() / future_values_mask.numel()
        masked_future_abs = future_values.abs() * future_values_mask
        target_abs_mean = masked_future_abs.sum() / future_values_mask.sum().clamp_min(1.0)
        target_abs_max = masked_future_abs.max()
        self._record_loss_components(
            phase,
            point_loss=point_loss,
            mse_loss=mse_loss,
            quantile_loss=quantile_loss,
            valid_future_ratio=valid_future_ratio,
            valid_series_ratio=valid_series.to(torch.float32).mean(),
            target_abs_mean=target_abs_mean,
            target_abs_max=target_abs_max,
        )

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._sanitize_nonfinite_gradients()
        return loss

    def _sanitize_nonfinite_gradients(self):
        model = self.model
        for param in model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)

    def _clip_grad_norm(self, model):
        pre_clip_norm = super()._clip_grad_norm(model)
        post_clip_norm = self.accelerator.clip_grad_norm_(model.parameters(), float("inf"))
        self._last_grad_norm_pre_clip = pre_clip_norm
        self._last_grad_norm_post_clip = post_clip_norm
        return pre_clip_norm

    def _get_grad_norm(self, model, grad_norm=None):
        post_clip_norm = getattr(self, "_last_grad_norm_post_clip", None)
        if post_clip_norm is not None:
            return post_clip_norm
        return super()._get_grad_norm(model, grad_norm=grad_norm)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        pre_clip_norm = getattr(self, "_last_grad_norm_pre_clip", None)
        if "grad_norm" in logs and pre_clip_norm is not None:
            logs["grad_norm_pre_clip"] = self._to_float(pre_clip_norm)
        if "loss" in logs:
            self._flush_loss_components(logs, "train")
        if "eval_loss" in logs:
            self._flush_loss_components(logs, "eval", prefix="eval_")
        super().log(logs, start_time=start_time)

    def floating_point_ops(self, inputs: dict) -> int:
        """Avoid Transformers' default tensor-only FLOPs estimate.

        TimesFM-2.5 receives `past_values` as a list of variable-length tensors,
        while the default Trainer assumes the main input has `.numel()`.
        Returning 0 only disables FLOPs accounting; it does not affect training.
        """

        return 0

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader_params = self._common_dataloader_params()
        return DataLoader(cast(IterableDataset, self.train_dataset), **dataloader_params)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        dataloader_params = self._common_dataloader_params()
        return DataLoader(cast(IterableDataset, dataset), **dataloader_params)

    def _common_dataloader_params(self) -> dict:
        params = {
            "batch_size": None,
            "collate_fn": None,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "worker_init_fn": seed_worker,
        }
        if self.args.dataloader_num_workers > 0:
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return params
