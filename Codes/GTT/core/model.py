import copy
import json
import logging
import os
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from core.network import GTTNet
from utils.data_handler import DataHandler
from utils.data_util import DataUtil

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    block_size: int = None
    patch_size: int = None
    target_dim: int = None
    covariate_dim: int = None
    timefeat_dim: int = None
    embedd_pdrop: float = 0.1
    dropout: float = 0.0
    activation_dropout: float = 0.0
    attention_dropout: float = 0.0
    n_embd: int = 768
    encoder_layers: int = 8
    encoder_attention_heads: int = 12
    encoder_layerdrop: float = 0.0
    encoder_ffn_dim: int = 3072
    enable_revin: bool = False
    affine: bool = False
    revin_time: bool = True
    forecast_mode: str = "point"
    pred_len: int = None
    point_loss: str = "huber"
    huber_beta: float = 1.0


class TrainHistory:
    def __init__(self):
        self.history = {"loss": [], "masked_mae": [], "val_loss": [], "val_masked_mae": []}


class GTT:
    def __init__(self, configs=None):
        self.configs = configs
        self._meta_init()

    def _meta_init(self):
        self.configs.target_dim = 24
        self.configs.covariate_dim = 0
        self.configs.timefeat_dim = 0
        if self.configs.pred_len is None:
            self._data_handler = DataHandler(self.configs.block_size, self.configs.patch_size)
        else:
            self._data_handler = DataHandler(self.configs.block_size, self.configs.pred_len)

    @property
    def data_handler(self):
        return self._data_handler

    @staticmethod
    def _unwrap_model(model):
        return model.module if isinstance(model, nn.DataParallel) else model

    @staticmethod
    def _move_optimizer_state_to_device(optimizer, device):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)

    @staticmethod
    def _masked_point_loss_and_mae(y_true, y_pred, point_loss="huber", huber_beta=1.0):
        f_max = 24
        eps = 1e-6
        y = y_true[:, :, :f_max]
        m = y_true[:, :, f_max:f_max * 2]
        finite_pred = torch.isfinite(y_pred)
        m = m * finite_pred.to(m.dtype)
        y_pred = torch.where(finite_pred, y_pred, torch.zeros_like(y_pred).detach())
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        diff = y_pred - y
        point_loss = str(point_loss).lower()
        if point_loss == "huber":
            loss_values = F.smooth_l1_loss(
                y_pred,
                y,
                reduction="none",
                beta=float(huber_beta),
            )
        elif point_loss == "mse":
            loss_values = torch.square(diff)
        else:
            raise ValueError(f"Unsupported point_loss: {point_loss}")
        ae = torch.abs(diff) * m
        denom = torch.sum(m).clamp_min(eps)
        loss = torch.sum(loss_values * m) / denom
        mae = torch.sum(ae) / denom
        return loss, mae

    def train(
        self,
        train_list,
        val_list,
        cp,
        pm=None,
        optimizer=None,
        batch_size=256,
        epochs=10,
        distribute=False,
        mixed_precision=False,
        verbose=0,
        save_steps=1000,
        val_steps=None,
        resume_backup=False,
        learning_rate=1e-4,
        weight_decay=0.0,
        grad_clip_norm=1.0,
        warmup_ratio=0.0,
        warmup_steps=300,
        min_lr_ratio=0.0,
        use_lr_schedule=True,
        lr_scheduler_type="linear",
        gradient_accumulation_steps=1,
        global_shuffle=True,
        shuffle_seed=0,
        shuffle_buffer_size=1_000_000,
        log_filename="train.log",
        log_file_mode="a",
        create_backup=True,
        resume_optimizer_lr=None,
        resume_epoch_offset=0,
        eval_milestone_ratios=None,
        save_on_eval=False,
    ):
        if not os.path.exists(cp):
            os.makedirs(cp)

        log_path = os.path.join(cp, log_filename)
        if not any(getattr(h, "baseFilename", None) == os.path.abspath(log_path) for h in logger.handlers):
            fh = logging.FileHandler(log_path, mode=log_file_mode, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.setLevel(logging.INFO)
            logger.addHandler(fh)
            logger.propagate = False

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        model = GTTNet.build_raw_model(self.configs).to(device)
        resume_checkpoint = None
        optimizer_resume_lrs = None
        resumed_optimizer = False
        if pm is not None:
            state = torch.load(pm, map_location=device)
            if isinstance(state, dict) and "model" in state:
                resume_checkpoint = state
                model.load_state_dict(state["model"])
                print(f"loaded full checkpoint model from {pm}")
            else:
                model.load_state_dict(state)
                print(f"loaded model weights from {pm}")

        total_para = sum(p.numel() for p in model.parameters())
        print("num_replicas_in_sync:", device_count if distribute and device_count > 0 else 1)
        print("para", total_para)
        print(
            f"model_norm=external_chronos enable_revin={self.configs.enable_revin} "
            f"affine={self.configs.affine} point_loss={self.configs.point_loss} "
            f"huber_beta={self.configs.huber_beta}"
        )

        if optimizer is None:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(learning_rate),
                weight_decay=float(weight_decay),
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        if resume_checkpoint is not None and resume_checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            self._move_optimizer_state_to_device(optimizer, device)
            optimizer_resume_lrs = [group["lr"] for group in optimizer.param_groups]
            resumed_optimizer = True
            print(f"loaded optimizer state from {pm}")
        if resumed_optimizer and resume_optimizer_lr is not None:
            resume_optimizer_lr = float(resume_optimizer_lr)
            for group in optimizer.param_groups:
                group["lr"] = resume_optimizer_lr
                group["initial_lr"] = resume_optimizer_lr
            optimizer_resume_lrs = [group["lr"] for group in optimizer.param_groups]
            print(f"override resumed optimizer lr to {resume_optimizer_lr}")
        if distribute and device_count > 1:
            model = nn.DataParallel(model)

        gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
        global_shuffle = bool(global_shuffle)
        shuffle_seed = int(shuffle_seed)
        shuffle_buffer_size = int(shuffle_buffer_size)

        global_batch_size = (device_count if distribute and device_count > 0 else 1) * batch_size
        effective_batch_size = global_batch_size * gradient_accumulation_steps
        steps_per_epoch = max(1, len(train_list) // global_batch_size)
        update_steps_per_epoch = max(1, int(np.ceil(steps_per_epoch / gradient_accumulation_steps)))
        val_steps_full = len(val_list) // global_batch_size
        eval_val_steps = val_steps_full
        if val_steps is not None:
            eval_val_steps = max(0, min(val_steps_full, int(val_steps)))
        print(
            f"train_count={len(train_list)} val_count={len(val_list)} "
            f"global_batch_size={global_batch_size} steps_per_epoch={steps_per_epoch} "
            f"gradient_accumulation_steps={gradient_accumulation_steps} "
            f"effective_batch_size={effective_batch_size} update_steps_per_epoch={update_steps_per_epoch} "
            f"val_steps={eval_val_steps} epochs={epochs} "
            f"global_shuffle={global_shuffle} shuffle_seed={shuffle_seed} "
            f"shuffle_buffer_size={shuffle_buffer_size}"
        )
        print(
            f"optimizer={optimizer.__class__.__name__} lr={learning_rate} "
            f"weight_decay={weight_decay} grad_clip_norm={grad_clip_norm} "
            f"warmup_steps={warmup_steps} warmup_ratio={warmup_ratio} "
            f"min_lr_ratio={min_lr_ratio} use_lr_schedule={use_lr_schedule} "
            f"lr_scheduler_type={lr_scheduler_type} "
            f"point_loss={self.configs.point_loss} huber_beta={self.configs.huber_beta}"
        )
        logger.info(
            f"train_count={len(train_list)} val_count={len(val_list)} "
            f"global_batch_size={global_batch_size} steps_per_epoch={steps_per_epoch} "
            f"gradient_accumulation_steps={gradient_accumulation_steps} "
            f"effective_batch_size={effective_batch_size} update_steps_per_epoch={update_steps_per_epoch} "
            f"val_steps={eval_val_steps} epochs={epochs} "
            f"optimizer={optimizer.__class__.__name__} lr={learning_rate} "
            f"weight_decay={weight_decay} grad_clip_norm={grad_clip_norm} "
            f"warmup_steps={warmup_steps} warmup_ratio={warmup_ratio} "
            f"min_lr_ratio={min_lr_ratio} use_lr_schedule={use_lr_schedule} "
            f"lr_scheduler_type={lr_scheduler_type} "
            f"global_shuffle={global_shuffle} shuffle_seed={shuffle_seed} "
            f"shuffle_buffer_size={shuffle_buffer_size} "
            f"point_loss={self.configs.point_loss} huber_beta={self.configs.huber_beta}"
        )

        if create_backup:
            backup_dir = os.path.join(cp, "backup" if resume_backup else f"backup_run_{os.getpid()}")
            if not resume_backup and os.path.exists(os.path.join(cp, "backup")):
                logger.info("Found existing backup directory but resume_backup=False; using a fresh backup directory.")
                print(
                    f"Found existing backup at {os.path.join(cp, 'backup')}; "
                    "ignore it for this run. Use resume_backup=True if you really want to resume."
                )
            os.makedirs(backup_dir, exist_ok=True)

        val_loader = self._data_handler.make_dataset(val_list, global_batch_size, training=False, shuffle=False)

        history = TrainHistory()
        best_val_loss = None
        best_state = None
        best_optimizer_state = None
        best_scheduler_state = None
        best_scaler_state = None
        best_global_step = None
        patience = 3
        wait = 0
        use_lr_schedule = bool(use_lr_schedule)
        lr_scheduler_type = str(lr_scheduler_type).lower()
        if lr_scheduler_type not in {"linear", "cosine", "constant"}:
            raise ValueError(f"Unsupported lr_scheduler_type: {lr_scheduler_type}")
        total_steps = update_steps_per_epoch * int(epochs)
        if warmup_steps is None:
            warmup_steps = int(total_steps * float(warmup_ratio))
        else:
            warmup_steps = int(warmup_steps)
        min_lr_ratio = float(min_lr_ratio)

        def lr_lambda(step):
            if total_steps <= 0:
                return 1.0
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            if lr_scheduler_type == "constant":
                return 1.0
            progress_steps = max(1, total_steps - warmup_steps)
            progress = min(1.0, max(0.0, (step - warmup_steps) / progress_steps))
            if lr_scheduler_type == "linear":
                decay = 1.0 - progress
            else:
                decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * decay

        checkpoint_has_scheduler = resume_checkpoint is not None and resume_checkpoint.get("scheduler") is not None
        if use_lr_schedule and resumed_optimizer and not checkpoint_has_scheduler:
            msg = (
                "full checkpoint has optimizer but no scheduler; "
                "starting a fresh resume lr schedule from the overridden optimizer lr"
            )
            print(msg)
            logger.info(msg)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda) if use_lr_schedule else None
        if scheduler is not None and checkpoint_has_scheduler:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
            if optimizer_resume_lrs is not None:
                for param_group, lr in zip(optimizer.param_groups, optimizer_resume_lrs):
                    param_group["lr"] = lr
            print(f"loaded scheduler state from {pm}")
        global_step = int(resume_checkpoint.get("global_step", 0) or 0) if resume_checkpoint is not None else 0
        completed_epochs = (
            int(resume_checkpoint.get("epochs_completed", 0) or 0) if resume_checkpoint is not None else 0
        )
        if resume_checkpoint is not None and "epochs_completed" not in resume_checkpoint:
            completed_epochs = int(resume_epoch_offset)
        auto_save_steps = set() if save_on_eval else {
            global_step + max(1, int(np.ceil(total_steps * ratio / 2.0))) for ratio in (1, 2)
        }
        if eval_milestone_ratios is None:
            eval_milestone_ratios = (0.15, 0.35, 0.75, 1.0)
        eval_milestones = [
            (ratio, min(steps_per_epoch, max(1, int(np.ceil(steps_per_epoch * ratio)))))
            for ratio in eval_milestone_ratios
        ]
        eval_schedule_msg = "eval_schedule=" + ",".join(
            f"{ratio:.0%}@batch{trigger_batch}" for ratio, trigger_batch in eval_milestones
        )
        print(eval_schedule_msg)
        logger.info(eval_schedule_msg)
        amp_enabled = bool(mixed_precision and torch.cuda.is_available())
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        if resume_checkpoint is not None and resume_checkpoint.get("scaler") is not None:
            scaler.load_state_dict(resume_checkpoint["scaler"])
            print(f"loaded grad scaler state from {pm}")

        def autocast_context():
            try:
                return torch.amp.autocast("cuda", enabled=amp_enabled)
            except (AttributeError, TypeError):
                return torch.cuda.amp.autocast(enabled=amp_enabled)

        def save_eval_checkpoint(epoch_percent, val_loss, val_mae, step, epochs_completed_value=None):
            pct = int(round(float(epoch_percent) * 100))
            checkpoint_path = os.path.join(cp, f"checkpoint_eval{pct:03d}.pt")
            saved_epochs_completed = completed_epochs if epochs_completed_value is None else epochs_completed_value
            torch.save(
                {
                    "model": self._unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "global_step": int(step),
                    "epochs_completed": int(saved_epochs_completed),
                    "epoch_percent": float(epoch_percent),
                    "val_loss": float(val_loss),
                    "val_masked_mae": float(val_mae),
                    "config": asdict(self.configs),
                },
                checkpoint_path,
            )
            logger.info(f"[step {step}] saved_full_eval_checkpoint={checkpoint_path}")

        def update_best(val_loss, step):
            nonlocal best_val_loss, best_state, best_optimizer_state, best_scheduler_state, best_scaler_state
            nonlocal best_global_step, wait
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self._unwrap_model(model).state_dict().items()}
                best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                best_scheduler_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
                best_scaler_state = copy.deepcopy(scaler.state_dict()) if scaler is not None else None
                best_global_step = int(step)
                torch.save(best_state, os.path.join(cp, "GTT_best.pt"))
                logger.info(f"[step {step}] best_val_loss={best_val_loss} saved_weights={os.path.join(cp, 'GTT_best.pt')}")
                wait = 0
                return True
            wait += 1
            return False

        def grad_total_norm(parameters):
            grads = [p.grad.detach() for p in parameters if p.grad is not None]
            if not grads:
                return 0.0
            device = grads[0].device
            norms = torch.stack([torch.linalg.vector_norm(g, ord=2).to(device) for g in grads])
            return float(torch.linalg.vector_norm(norms, ord=2).detach().cpu())

        for epoch in range(int(epochs)):
            absolute_epoch = completed_epochs + epoch
            epoch_seed = shuffle_seed + absolute_epoch
            if global_shuffle:
                random.Random(epoch_seed).shuffle(train_list)
                shuffle_msg = (
                    f"[epoch {epoch + 1}] global_shuffle=True in_place=True "
                    f"absolute_epoch={absolute_epoch + 1} "
                    f"seed={epoch_seed} rows={len(train_list)}"
                )
                logger.info(shuffle_msg)
                if verbose:
                    print(shuffle_msg)

            train_loader = self._data_handler.make_dataset(
                train_list,
                global_batch_size,
                training=True,
                seed=epoch_seed,
                shuffle_buffer_size=shuffle_buffer_size,
                shuffle=not global_shuffle,
            )
            train_iter = iter(train_loader)
            model.train()
            epoch_loss = []
            epoch_mae = []
            mid_val_done = set()
            pbar = tqdm(
                range(steps_per_epoch),
                total=steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{epochs}",
                dynamic_ncols=True,
                mininterval=1.0,
            )
            for batch_idx in pbar:
                x, y_pack = next(train_iter)
                x = x.to(device, non_blocking=True).float()
                y_pack = y_pack.to(device, non_blocking=True).float()
                if batch_idx % gradient_accumulation_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                    accumulation_window_steps = min(gradient_accumulation_steps, steps_per_epoch - batch_idx)
                with autocast_context():
                    y_pred = model(x)
                    loss, mae = self._masked_point_loss_and_mae(
                        y_pack,
                        y_pred,
                        point_loss=self.configs.point_loss,
                        huber_beta=self.configs.huber_beta,
                    )
                    scaled_loss = loss / accumulation_window_steps
                scaler.scale(scaled_loss).backward()
                grad_norm_before = None
                grad_norm_after = None
                did_optimizer_step = (
                    (batch_idx + 1) % gradient_accumulation_steps == 0
                    or (batch_idx + 1) == steps_per_epoch
                )
                if did_optimizer_step:
                    next_global_step = global_step + 1
                    log_this_step = next_global_step % 10 == 0
                    should_clip_grad = grad_clip_norm is not None and float(grad_clip_norm) > 0
                    should_measure_grad = should_clip_grad or log_this_step
                    if should_measure_grad:
                        scaler.unscale_(optimizer)
                    if should_clip_grad:
                        grad_norm_before_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                        grad_norm_before = float(grad_norm_before_tensor.detach().cpu())
                        if log_this_step:
                            grad_norm_after = grad_total_norm(model.parameters())
                    elif log_this_step:
                        grad_norm_before = grad_total_norm(model.parameters())
                        grad_norm_after = grad_norm_before
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    global_step += 1
                epoch_loss.append(float(loss.detach().cpu()))
                epoch_mae.append(float(mae.detach().cpu()))

                if did_optimizer_step and global_step % 10 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    avg_loss_100 = float(np.mean(epoch_loss[-100:]))
                    avg_mae_100 = float(np.mean(epoch_mae[-100:]))
                    logger.info(
                        f"[step {global_step}] train_logs={{'loss': {epoch_loss[-1]}, "
                        f"'masked_mae': {epoch_mae[-1]}, "
                        f"'avg_loss_100': {avg_loss_100}, "
                        f"'avg_masked_mae_100': {avg_mae_100}, "
                        f"'lr': {current_lr}, "
                        f"'grad_norm_before_clip': {grad_norm_before}, "
                        f"'grad_norm_after_clip': {grad_norm_after}, "
                        f"'batch_index': {batch_idx + 1}, "
                        f"'batches_per_epoch': {steps_per_epoch}, "
                        f"'per_device_batch_size': {batch_size}, "
                        f"'global_batch_size': {global_batch_size}, "
                        f"'gradient_accumulation_steps': {gradient_accumulation_steps}, "
                        f"'effective_batch_size': {effective_batch_size}}}"
                    )
                    pbar.write(
                        f"[step {global_step}] loss={epoch_loss[-1]:.6f} "
                        f"masked_mae={epoch_mae[-1]:.6f} "
                        f"avg_loss_100={avg_loss_100:.6f} avg_mae_100={avg_mae_100:.6f} "
                        f"lr={current_lr:.8g} "
                        f"batch={batch_idx + 1}/{steps_per_epoch} "
                        f"global_batch={global_batch_size} effective_batch={effective_batch_size} "
                        f"grad_before_clip={grad_norm_before} grad_after_clip={grad_norm_after}"
                    )
                pbar.set_postfix(
                    loss=f"{epoch_loss[-1]:.6f}",
                    masked_mae=f"{epoch_mae[-1]:.6f}",
                    avg100=f"{float(np.mean(epoch_loss[-100:])):.6f}",
                    global_step=global_step,
                    batch=f"{batch_idx + 1}/{steps_per_epoch}",
                    accum=f"{(batch_idx % gradient_accumulation_steps) + 1}/{accumulation_window_steps}",
                    grad_before=None if grad_norm_before is None else f"{grad_norm_before:.4f}",
                    grad_after=None if grad_norm_after is None else f"{grad_norm_after:.4f}",
                )

                if did_optimizer_step and (
                    global_step in auto_save_steps or (save_steps > 0 and global_step % int(save_steps) == 0)
                ):
                    prefix = os.path.join(cp, f"GTT_step{global_step:06d}.pt")
                    torch.save(self._unwrap_model(model).state_dict(), prefix)
                    logger.info(f"[step {global_step}] saved_weights_prefix={prefix}")

                if did_optimizer_step and eval_val_steps > 0 and (batch_idx + 1) < steps_per_epoch:
                    pending_eval = None
                    for ratio, trigger_batch in eval_milestones:
                        if ratio not in mid_val_done and (batch_idx + 1) >= trigger_batch:
                            pending_eval = ratio
                            break
                    if pending_eval is not None:
                        mid_val_done.add(pending_eval)
                        pbar.write(f"[step {global_step}] running {pending_eval:.0%} validation...")
                        val_loss, val_mae = self._evaluate(model, val_loader, eval_val_steps, device)
                        logger.info(
                            f"[step {global_step}] interim_val_result={{'epoch_percent': {pending_eval}, "
                            f"'val_loss': {val_loss}, 'val_masked_mae': {val_mae}}}"
                        )
                        update_best(val_loss, global_step)
                        if save_on_eval:
                            save_eval_checkpoint(pending_eval, val_loss, val_mae, global_step)
                        pbar.write(
                            f"[step {global_step}] {pending_eval:.0%} val_loss: {val_loss:.6f} "
                            f"- val_masked_mae: {val_mae:.6f}"
                        )

            history.history["loss"].append(float(np.mean(epoch_loss)))
            history.history["masked_mae"].append(float(np.mean(epoch_mae)))

            if eval_val_steps > 0:
                val_loss, val_mae = self._evaluate(model, val_loader, eval_val_steps, device)
                history.history["val_loss"].append(val_loss)
                history.history["val_masked_mae"].append(val_mae)
                logger.info(
                    f"[step {global_step}] final_val_result={{'epoch_percent': 1.0, "
                    f"'val_loss': {val_loss}, 'val_masked_mae': {val_mae}}}"
                )
                tqdm.write(f"100% val_loss: {val_loss:.6f} - val_masked_mae: {val_mae:.6f}")
                update_best(val_loss, global_step)
                if save_on_eval:
                    save_eval_checkpoint(1.0, val_loss, val_mae, global_step, absolute_epoch + 1)
                if wait >= patience:
                    completed_epochs = absolute_epoch + 1
                    break
            completed_epochs = absolute_epoch + 1

        if best_state is not None:
            self._unwrap_model(model).load_state_dict(best_state)
            if best_optimizer_state is not None:
                optimizer.load_state_dict(best_optimizer_state)
                self._move_optimizer_state_to_device(optimizer, device)
            if scheduler is not None and best_scheduler_state is not None:
                scheduler.load_state_dict(best_scheduler_state)
            if best_scaler_state is not None:
                scaler.load_state_dict(best_scaler_state)
            if best_global_step is not None:
                global_step = best_global_step

        self.estimator = self._unwrap_model(model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.global_step = global_step
        self.epochs_completed = completed_epochs
        return history

    def _evaluate(self, model, val_loader, val_steps, device):
        model.eval()
        losses = []
        maes = []
        val_iter = iter(val_loader)
        with torch.no_grad():
            for _ in range(int(val_steps)):
                x, y_pack = next(val_iter)
                x = x.to(device, non_blocking=True).float()
                y_pack = y_pack.to(device, non_blocking=True).float()
                y_pred = model(x)
                loss, mae = self._masked_point_loss_and_mae(
                    y_pack,
                    y_pred,
                    point_loss=self.configs.point_loss,
                    huber_beta=self.configs.huber_beta,
                )
                losses.append(float(loss.detach().cpu()))
                maes.append(float(mae.detach().cpu()))
        model.train()
        return float(np.mean(losses)), float(np.mean(maes))

    def predict_index(self, index_row):
        x, y, mask, mean, std, y_raw = self._data_handler.index_to_array_with_stats(index_row, training=False)
        device = next(self.estimator.parameters()).device
        self.estimator.eval()
        with torch.no_grad():
            y_pred = self.estimator(torch.from_numpy(x[None, :, :]).to(device).float()).cpu().numpy()[0]
        y_pred = self._data_handler.denormalize(y_pred, mean, std)
        return y_pred, y_raw, mask

    def save_model(self, model_path=None, hist=None, training_config=None):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.estimator.state_dict(), os.path.join(model_path, "GTT.pt"))
        pickle.dump(asdict(self.configs), open(os.path.join(model_path, "configs.pkl"), "wb"))
        ckpt_dir = os.path.join(model_path, "checkpoint-final")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {
                "model": self.estimator.state_dict(),
                "optimizer": self.optimizer.state_dict() if hasattr(self, "optimizer") and self.optimizer is not None else None,
                "scheduler": self.scheduler.state_dict() if hasattr(self, "scheduler") and self.scheduler is not None else None,
                "scaler": self.scaler.state_dict() if hasattr(self, "scaler") and self.scaler is not None else None,
                "global_step": getattr(self, "global_step", None),
                "epochs_completed": getattr(self, "epochs_completed", None),
                "config": asdict(self.configs),
                "training_config": training_config or {},
                "history": hist.history if hist is not None else None,
            },
            os.path.join(ckpt_dir, "checkpoint.pt"),
        )
        training_info = {
            "model_class": "GTT",
            "weights_file": "GTT.pt",
            "checkpoint_file": "checkpoint-final/checkpoint.pt",
            "config": asdict(self.configs),
            "data_format": {
                "index_row": ["root", "dataset", "numpy", "time_start", "time_stop", "target", "cov"],
                "data_path_template": "{root}/Data/{dataset}/{numpy}.npy",
                "target_cov_order": "target columns first, covariate columns after target columns",
                "max_channels": 24,
                "x_shape": [self.configs.block_size, 24],
                "y_pack": "first 24 channels are normalized values; last 24 channels are loss mask",
                "prediction_output": "predict_index returns predictions and ground truth in original value space",
                "normalization": "Chronos-style: loc/scale from context, arcsinh, clip [-5, 5], inverse uses sinh",
                "point_loss": self.configs.point_loss,
                "huber_beta": self.configs.huber_beta,
            },
            "training_config": training_config or {},
        }
        with open(os.path.join(model_path, "training_info.json"), "w", encoding="utf-8") as fp:
            json.dump(training_info, fp, indent=4)
        if hist is not None:
            pickle.dump(hist.history, open(os.path.join(model_path, "history.pkl"), "wb"))

    def load_model(self, model_path=None, pm=None):
        configs = pickle.load(open(os.path.join(model_path, "configs.pkl"), "rb"))
        self.configs = ModelConfig(**configs)
        data_util_path = os.path.join(model_path, "data_util.json")
        self.du = DataUtil(filename=data_util_path) if os.path.exists(data_util_path) else None
        self._meta_init()
        model = GTTNet.build_raw_model(self.configs)
        state = torch.load(os.path.join(model_path, "GTT.pt"), map_location="cpu")
        model.load_state_dict(state)
        self.estimator = model
        return self


class TSFoundation:
    def __init__(self, configs=None):
        self.configs = configs
