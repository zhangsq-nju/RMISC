import json
import random
from pathlib import Path
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf


class PretrainStateCheckpoint(Callback):
    def __init__(
        self,
        dirpath: str | Path,
        cfg: DictConfig,
        training_args: dict[str, Any],
        every_n_train_steps: int,
        save_last: bool = True,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.cfg = cfg
        self.training_args = training_args
        self.every_n_train_steps = every_n_train_steps
        self.save_last = save_last
        self._last_global_step_saved = -1

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step = trainer.global_step
        if step <= 0 or step == self._last_global_step_saved:
            return
        if step % self.every_n_train_steps != 0:
            return
        self._save_checkpoint_dir(trainer, pl_module, self.dirpath / f"checkpoint-{step}")
        self._last_global_step_saved = step

    def on_train_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        if not self.save_last:
            return
        self._save_checkpoint_dir(trainer, pl_module, self.dirpath / "checkpoint-last")
        self._last_global_step_saved = trainer.global_step

    def _save_checkpoint_dir(
        self, trainer: L.Trainer, pl_module: L.LightningModule, checkpoint_dir: Path
    ) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        rank = int(getattr(trainer, "global_rank", 0))
        self._save_rng_state(checkpoint_dir / f"rng_state_{rank}.pth")

        if hasattr(trainer.strategy, "barrier"):
            trainer.strategy.barrier("pretrain_state_checkpoint_rng")

        if trainer.is_global_zero:
            ### 修改了checkpoint保存格式：按用户指定文件名保存可断点续训状态和safetensors模型权重。
            self._save_safetensors(
                self._state_dict_for_safetensors(pl_module),
                checkpoint_dir / "model.safetensors",
            )
            torch.save(self._optimizer_state(trainer), checkpoint_dir / "optimizer.pt")
            torch.save(self._scheduler_state(trainer), checkpoint_dir / "scheduler.pt")
            torch.save(self.training_args, checkpoint_dir / "training_args.bin")
            self._write_json(checkpoint_dir / "config.json", self._config_dict())
            self._write_json(
                checkpoint_dir / "trainer_state.json",
                self._trainer_state_dict(trainer),
            )

        if hasattr(trainer.strategy, "barrier"):
            trainer.strategy.barrier("pretrain_state_checkpoint_done")

    @staticmethod
    def _state_dict_for_safetensors(
        pl_module: L.LightningModule,
    ) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu()
            for key, value in pl_module.state_dict().items()
            if isinstance(value, torch.Tensor)
        }

    @staticmethod
    def _save_safetensors(state_dict: dict[str, torch.Tensor], path: Path) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError as error:
            raise RuntimeError(
                "safetensors is required to save model.safetensors. "
                "Install the project dependencies or install safetensors."
            ) from error

        save_file(state_dict, path)

    @staticmethod
    def _optimizer_state(trainer: L.Trainer) -> Any:
        if len(trainer.optimizers) == 1:
            return trainer.optimizers[0].state_dict()
        return [optimizer.state_dict() for optimizer in trainer.optimizers]

    @staticmethod
    def _scheduler_state(trainer: L.Trainer) -> Any:
        schedulers = [
            config.scheduler.state_dict()
            for config in trainer.lr_scheduler_configs
            if config.scheduler is not None
        ]
        return schedulers[0] if len(schedulers) == 1 else schedulers

    def _config_dict(self) -> dict[str, Any]:
        return OmegaConf.to_container(self.cfg, resolve=True)

    @staticmethod
    def _trainer_state_dict(trainer: L.Trainer) -> dict[str, Any]:
        callbacks = {}
        for callback in trainer.callbacks:
            state_key = getattr(callback, "state_key", callback.__class__.__qualname__)
            state = callback.state_dict()
            if state:
                callbacks[state_key] = state

        return {
            "global_step": trainer.global_step,
            "current_epoch": trainer.current_epoch,
            "max_epochs": trainer.max_epochs,
            "max_steps": trainer.max_steps,
            "world_size": trainer.world_size,
            "global_rank": trainer.global_rank,
            "callback_states": callbacks,
            "logged_metrics": {
                key: PretrainStateCheckpoint._json_value(value)
                for key, value in trainer.logged_metrics.items()
            },
            "callback_metrics": {
                key: PretrainStateCheckpoint._json_value(value)
                for key, value in trainer.callback_metrics.items()
            },
        }

    @staticmethod
    def _save_rng_state(path: Path) -> None:
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else [],
        }
        torch.save(rng_state, path)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _json_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.detach().cpu().item()
            return value.detach().cpu().tolist()
        return value
