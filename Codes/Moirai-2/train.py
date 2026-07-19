#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
import ast
import logging
import math
import os
import re
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import hydra
import lightning as L
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from torch.utils._pytree import tree_map
from torch.utils.data import Dataset, DistributedSampler, IterableDataset
from tqdm.auto import tqdm

from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.callbacks.PretrainStateCheckpoint import PretrainStateCheckpoint
from uni2ts.data.builder.indexed_npy import IndexedNpyDatasetBuilder
from uni2ts.data.loader import DataLoader


def _format_training_data_tag(training_data_paths: str) -> str:
    raw_paths = re.split(r"[,+]", training_data_paths)
    names = []
    for raw_path in raw_paths:
        raw_path = raw_path.strip().rstrip("/\\")
        if not raw_path:
            continue
        name = Path(raw_path).name or raw_path
        name = re.sub(r"[^0-9A-Za-z_-]+", "", name)
        if name:
            names.append(name)
    if not names:
        raise ValueError("--training-data-paths cannot be empty")
    return "".join(names)


def _parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--training-data-paths", required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    args, hydra_args = parser.parse_known_args()
    if args.num_epochs <= 0:
        raise ValueError("--num-epochs must be greater than 0")
    os.environ["UNI2TS_TRAINING_DATA_TAG"] = _format_training_data_tag(
        args.training_data_paths
    )
    sys.argv = [sys.argv[0], *hydra_args]
    return args


ARGS = _parse_train_args()


def _get_env_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _get_runtime_world_size(trainer: Optional[L.Trainer] = None) -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return max(1, torch.distributed.get_world_size())
    world_size = _get_env_int("WORLD_SIZE")
    if world_size is not None:
        return world_size
    if trainer is not None:
        return max(1, int(getattr(trainer, "world_size", 1)))
    return 1


def _get_runtime_global_rank(trainer: Optional[L.Trainer] = None) -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return max(0, torch.distributed.get_rank())
    rank = _get_env_int("RANK")
    if rank is not None:
        return rank
    if trainer is not None:
        return max(0, int(getattr(trainer, "global_rank", 0)))
    return 0


def _get_runtime_local_world_size() -> int:
    return _get_env_int("LOCAL_WORLD_SIZE") or _get_env_int("WORLD_SIZE") or 1


def _use_runtime_devices(cfg: DictConfig) -> None:
    if str(cfg.trainer.get("devices", "auto")) != "auto":
        return
    with open_dict(cfg):
        cfg.trainer.devices = _get_runtime_local_world_size()


class _TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:
            self.handleError(record)


def _setup_train_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("uni2ts.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = _TqdmLoggingHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_data_paths(training_data_paths: str) -> list[str]:
    ### 修改了数据路径解析：支持用逗号或加号合并多个数据root。
    raw_paths = re.split(r"[,+]", training_data_paths)
    data_paths = []
    for raw_path in raw_paths:
        raw_path = raw_path.strip()
        if not raw_path:
            continue

        candidates = [
            Path(raw_path),
            Path("dataset") / raw_path,
            Path("data") / raw_path,
            Path("dataset") / Path(raw_path).name,
        ]
        data_path = next((candidate for candidate in candidates if candidate.exists()), Path(raw_path))
        data_path = data_path.resolve()
        if not (data_path / "Data").is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_path / 'Data'}")
        if not (data_path / "DataIndex").is_dir():
            raise FileNotFoundError(f"DataIndex directory not found: {data_path / 'DataIndex'}")
        data_paths.append(str(data_path))

    if len(data_paths) == 0:
        raise ValueError("--training-data-paths cannot be empty")
    return data_paths


def _read_index_file(file_path: str, data_root: str) -> list[tuple]:
    df = pd.read_csv(
        file_path,
        usecols=["dataset", "numpy", "time_start", "time_stop", "target"],
    )

    rows = []
    for row in df.itertuples(index=False):
        target = ast.literal_eval(row.target) if isinstance(row.target, str) else row.target
        if not isinstance(target, (list, tuple)):
            target = [target]
        for target_col in target:
            rows.append(
                (
                    str(data_root),
                    str(row.dataset),
                    int(row.numpy),
                    int(row.time_start),
                    int(row.time_stop),
                    [target_col],
                )
            )
    return rows


def _infer_dataset_names(data_path: Path, split: str) -> list[str]:
    index_dir = data_path / "DataIndex" / split
    if not index_dir.is_dir():
        raise FileNotFoundError(f"DataIndex split directory not found: {index_dir}")
    dataset_names = sorted(path.stem for path in index_dir.glob("*.csv"))
    if not dataset_names:
        raise FileNotFoundError(f"No DataIndex csv files found in: {index_dir}")
    return dataset_names


def _build_indexed_npy_builder(
    template_cfg: DictConfig,
    data_path: Path,
    split: str,
    dataset: str,
    index_rows: list[tuple],
) -> IndexedNpyDatasetBuilder:
    return IndexedNpyDatasetBuilder(
        dataset=dataset,
        split=split,
        root_path=data_path,
        shuffle=template_cfg.shuffle,
        index_rows=index_rows,
        distance=template_cfg.distance,
        prediction_length=template_cfg.prediction_length,
        context_length=template_cfg.context_length,
        patch_size=template_cfg.patch_size,
    )


def _set_step_based_trainer_schedule(
    cfg: DictConfig,
    trainer: L.Trainer,
    total_train_rows: int,
) -> int:
    world_size = _get_runtime_world_size(trainer)
    accumulate = max(1, trainer.accumulate_grad_batches)
    batch_size_per_rank = cfg.train_dataloader.batch_size_per_rank
    steps_per_epoch = max(
        1,
        math.ceil(total_train_rows / (world_size * batch_size_per_rank * accumulate)),
    )
    eval_save_steps = max(1, math.ceil(steps_per_epoch / 2))

    trainer.val_check_interval = eval_save_steps * accumulate
    for callback in trainer.callbacks:
        if hasattr(callback, "_every_n_train_steps"):
            callback._every_n_train_steps = eval_save_steps
        if hasattr(callback, "_every_n_epochs"):
            callback._every_n_epochs = 0
    return steps_per_epoch


def _disable_lightning_progress_bar(trainer: L.Trainer) -> None:
    progress_bar_names = {"TQDMProgressBar", "RichProgressBar"}
    trainer.callbacks = [
        callback
        for callback in trainer.callbacks
        if callback.__class__.__name__ not in progress_bar_names
    ]


class TrainProgressLogger(L.Callback):
    def __init__(
        self,
        logger: logging.Logger,
        total_steps: int,
        steps_per_epoch: int,
        train_loss_metric: str,
        log_every_n_steps: int = 100,
    ):
        super().__init__()
        self.logger = logger
        self.total_steps = total_steps
        self.steps_per_epoch = steps_per_epoch
        self.train_loss_metric = train_loss_metric
        self.log_every_n_steps = log_every_n_steps
        self.start_time: Optional[float] = None
        self.last_logged_step = -1
        self.last_progress_step = -1
        self.last_eval_loss: Optional[float] = None
        self.last_grad_norm: Optional[float] = None
        self.progress_bar: Optional[tqdm] = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.start_time = time.time()
        self.last_progress_step = trainer.global_step
        if trainer.is_global_zero:
            self.progress_bar = tqdm(
                total=self.total_steps,
                initial=trainer.global_step,
                desc="Train",
                unit="opt_step",
                dynamic_ncols=True,
            )

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    def on_exception(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        exception: BaseException,
    ) -> None:
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None

    def on_before_optimizer_step(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        next_step = trainer.global_step + 1
        if next_step % self.log_every_n_steps != 0:
            return

        squared_norm = torch.tensor(0.0, device=pl_module.device)
        for parameter in pl_module.parameters():
            if parameter.grad is not None:
                squared_norm += parameter.grad.detach().data.norm(2).pow(2)
        self.last_grad_norm = float(squared_norm.sqrt().cpu())

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not trainer.is_global_zero:
            return
        step = trainer.global_step
        self._update_progress_bar(trainer, step)
        if step <= 0 or step == self.last_logged_step or step % self.log_every_n_steps != 0:
            return

        train_loss = self._get_batch_loss(outputs)
        if train_loss is not None:
            train_loss *= max(1, trainer.accumulate_grad_batches)
        elapsed = time.time() - self.start_time if self.start_time is not None else 0.0
        eta = _format_eta(elapsed / step * max(self.total_steps - step, 0)) if step else "00:00:00"

        record = {
            "train_loss": train_loss,
            "grad_norm": self.last_grad_norm,
            "learning_rate": self._get_learning_rate(trainer),
            "optimizer_step": step,
            "total_optimizer_steps": self.total_steps,
            "eta": eta,
            "epoch": round(step / max(self.steps_per_epoch, 1), 4),
        }
        self.logger.info(record)
        self.last_logged_step = step

    def _get_batch_loss(self, outputs: Any) -> Optional[float]:
        if isinstance(outputs, torch.Tensor):
            return _as_float(outputs)
        if isinstance(outputs, dict):
            for key in ("loss", "train_loss"):
                if key in outputs:
                    return _as_float(outputs[key])
        return None

    def _get_learning_rate(self, trainer: L.Trainer) -> Optional[float]:
        if not trainer.optimizers:
            return None
        optimizer = trainer.optimizers[0]
        if not optimizer.param_groups:
            return None
        return float(optimizer.param_groups[0]["lr"])

    def _get_logged_train_loss(self, trainer: L.Trainer) -> Optional[float]:
        metric_keys = (
            self.train_loss_metric,
            f"{self.train_loss_metric}_step",
        )
        for metrics in (trainer.callback_metrics, trainer.logged_metrics):
            for key in metric_keys:
                if key in metrics:
                    return _as_float(metrics[key])
        return None

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return

        metric_items = {
            key: _as_float(value)
            for key, value in trainer.callback_metrics.items()
            if key.startswith("val/")
        }
        if not metric_items:
            return

        primary_metric = next(iter(metric_items.values()))
        record = {
            "eval_loss": primary_metric,
            "optimizer_step": trainer.global_step,
            "total_optimizer_steps": self.total_steps,
            "epoch": trainer.current_epoch,
            **metric_items,
        }
        self.last_eval_loss = primary_metric
        self._update_progress_bar(trainer, trainer.global_step)
        self.logger.info(record)

    def _update_progress_bar(self, trainer: L.Trainer, step: int) -> None:
        if self.progress_bar is None:
            return

        if step > self.last_progress_step:
            self.progress_bar.update(step - self.last_progress_step)
            self.last_progress_step = step


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset | list[Dataset]],
        steps_per_epoch: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = train_dataset
        self.steps_per_epoch = steps_per_epoch

        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_dataloader = self._val_dataloader

    @staticmethod
    def get_dataloader(
        dataset: Dataset,
        dataloader_func: Callable[..., DataLoader],
        shuffle: bool,
        world_size: int,
        global_rank: int,
        batch_size_per_rank: int,
        num_batches_per_epoch: Optional[int] = None,
    ) -> DataLoader:
        is_iterable = isinstance(dataset, IterableDataset)
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=shuffle,
                seed=0,
                drop_last=False,
            )
            if world_size > 1 and not is_iterable
            else None
        )
        return dataloader_func(
            dataset=dataset,
            shuffle=False if is_iterable else shuffle,
            sampler=sampler,
            batch_size_per_rank=batch_size_per_rank,
            num_batches_per_epoch=num_batches_per_epoch,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(
            self.train_dataset,
            instantiate(self.cfg.train_dataloader, _partial_=True),
            self.cfg.train_dataloader.shuffle,
            _get_runtime_world_size(self.trainer),
            _get_runtime_global_rank(self.trainer),
            self.train_batch_size,
            num_batches_per_epoch=self.train_num_batches_per_epoch,
        )

    def _val_dataloader(self) -> DataLoader | list[DataLoader]:
        return self.get_dataloader(
            self.val_dataset,
            instantiate(self.cfg.val_dataloader, _partial_=True),
            self.cfg.val_dataloader.shuffle,
            _get_runtime_world_size(self.trainer),
            _get_runtime_global_rank(self.trainer),
            self.val_batch_size,
            num_batches_per_epoch=None,
        )


    @property
    def train_batch_size(self) -> int:
        return self.cfg.train_dataloader.batch_size_per_rank

    @property
    def val_batch_size(self) -> int:
        return self.cfg.val_dataloader.batch_size_per_rank

    @property
    def train_num_batches_per_epoch(self) -> int:
        return self.steps_per_epoch * self.trainer.accumulate_grad_batches


@hydra.main(version_base="1.3", config_path="conf/pretrain", config_name="default.yaml")
def main(cfg: DictConfig):
    args = ARGS
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    train_logger = _setup_train_logger(output_dir)
    if cfg.tf32:
        assert cfg.trainer.precision == 32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    with open_dict(cfg):
        cfg.trainer.max_epochs = args.num_epochs
    _use_runtime_devices(cfg)

    trainer: L.Trainer = instantiate(cfg.trainer)
    _disable_lightning_progress_bar(trainer)

    data_paths = resolve_data_paths(args.training_data_paths)
    if trainer.is_global_zero:
        train_logger.info(f"Resolved training data roots: {data_paths}")
    train_rows = []
    val_rows = []
    train_builder_root = None
    val_builder_root = None
    train_builder_dataset = None
    val_builder_dataset = None
    total_train_rows = 0
    for raw_data_path in data_paths:
        data_path = Path(raw_data_path)
        for dataset in _infer_dataset_names(data_path, "train"):
            index_file = data_path / "DataIndex" / "train" / f"{dataset}.csv"
            rows = _read_index_file(str(index_file), str(data_path))
            if trainer.is_global_zero:
                train_logger.info(f"Loading train index: {index_file} rows={len(rows)}")
            total_train_rows += len(rows)
            train_rows.extend(rows)
            train_builder_root = train_builder_root or data_path
            train_builder_dataset = train_builder_dataset or dataset
        for dataset in _infer_dataset_names(data_path, "val"):
            index_file = data_path / "DataIndex" / "val" / f"{dataset}.csv"
            rows = _read_index_file(str(index_file), str(data_path))
            if trainer.is_global_zero:
                train_logger.info(f"Loading val index: {index_file} rows={len(rows)}")
            val_rows.extend(rows)
            val_builder_root = val_builder_root or data_path
            val_builder_dataset = val_builder_dataset or dataset
    if train_builder_root is None or train_builder_dataset is None:
        raise ValueError("No train DataIndex rows found.")
    if val_builder_root is None or val_builder_dataset is None:
        raise ValueError("No val DataIndex rows found.")

    steps_per_epoch = _set_step_based_trainer_schedule(cfg, trainer, total_train_rows)
    eval_save_steps = max(1, math.ceil(steps_per_epoch / 2))
    with open_dict(cfg):

        cfg.model.num_training_steps = steps_per_epoch * args.num_epochs
        cfg.model.num_warmup_steps = min(cfg.model.num_warmup_steps, 0.08 * cfg.model.num_training_steps)
        cfg.model.log_on_step = False

    if trainer.is_global_zero:
        train_logger.info(
            "Using epoch-based stopping: "
            f"num_epochs={args.num_epochs}, "
            f"total_train_rows={total_train_rows}, "
            f"world_size={_get_runtime_world_size(trainer)}, "
            f"batch_size_per_rank={cfg.train_dataloader.batch_size_per_rank}, "
            f"gradient_accumulation_steps={max(1, trainer.accumulate_grad_batches)}, "
            f"steps_per_epoch={steps_per_epoch}, "
            f"eval_steps={eval_save_steps}, "
            f"save_steps={eval_save_steps}, "
            f"num_steps={cfg.model.num_training_steps}"
        )
        train_logger.info("Initializing model")

    model: L.LightningModule = instantiate(cfg.model, _convert_="all")

    if "collate_fn" not in cfg.train_dataloader:
        model.seq_fields = model.seq_fields + ("sample_id",)

    if cfg.compile:
        model.module.compile(mode=cfg.compile)

    val_transform_map = getattr(model, "val_transform_map", model.train_transform_map)
    
    train_dataset = _build_indexed_npy_builder(
        cfg.train_data, train_builder_root, "train", train_builder_dataset, train_rows
    ).load_dataset(model.train_transform_map)
    val_dataset = _build_indexed_npy_builder(
        cfg.val_data, val_builder_root, "val", val_builder_dataset, val_rows
    ).load_dataset(val_transform_map)

    L.seed_everything(cfg.seed + trainer.logger.version, workers=True)
    trainer.callbacks.append(
        TrainProgressLogger(
            logger=train_logger,
            total_steps=cfg.model.num_training_steps,
            steps_per_epoch=steps_per_epoch,
            train_loss_metric=f"train/{model.hparams.loss_func.__class__.__name__}",
            log_every_n_steps=10,
        )
    )
    trainer.callbacks.append(
        PretrainStateCheckpoint(
            dirpath=output_dir / "checkpoints",
            cfg=cfg,
            training_args=vars(args),
            every_n_train_steps=eval_save_steps,
            save_last=True,
        )
    )

    trainer.fit(
        model,
        datamodule=DataModule(cfg, train_dataset, val_dataset, steps_per_epoch),
        ckpt_path=cfg.ckpt_path,
    )

    print("Finished training!")


if __name__ == "__main__":
    main()




# train with dataset A
# torchrun --nproc_per_node=8 train.py --training-data-paths ../dataset/A --num-epochs 1

# train with datasets A+B
# torchrun --nproc_per_node=8 train.py --training-data-paths ../dataset/A+../dataset/B --num-epochs 1

