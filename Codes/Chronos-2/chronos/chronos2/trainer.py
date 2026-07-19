# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Authors: Abdul Fatir Ansari <ansarnd@amazon.com>

import warnings
from typing import TYPE_CHECKING, cast

from torch.utils.data import DataLoader, Dataset
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback

if TYPE_CHECKING:
    from chronos.chronos2.dataset import Chronos2Dataset


def seed_worker(worker_id: int):
    import random

    import numpy as np
    import torch

    seed = torch.initial_seed() % 2**32 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EvaluateAndSaveFinalStepCallback(TrainerCallback):
    """Callback to evaluate and save the model at last training step."""

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= state.max_steps:
            control.should_log = True
            control.should_evaluate = True
            control.should_save = True


class Chronos2Trainer(Trainer):
    """
    A custom trainer based on transformers Trainer. We need to override the dataloader getters because we handle
    batching ourselves in a custom dataset which directly returns batches instead of individual elements.
    """

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = cast("Chronos2Dataset", self.train_dataset)

        if self.args.train_batch_size > train_dataset.batch_size:
            warnings.warn(
                f"The batch_size of the train_dataset ({train_dataset.batch_size}) does not match the batch_size "
                f"in TrainingArguments ({self.args.train_batch_size}). On machines with multiple GPUs, this may indicate "
                f"that multiple GPUs are visible and transformers is using DataParallel for training by default. "
                f"This may lead to unnecessary slowdown and unexpected behavior. We strongly recommend setting the CUDA_VISIBLE_DEVICES "
                f"environment variable to ensure that only a single GPU is visible.",
                category=UserWarning,
                stacklevel=3,
            )

        dataloader_params = {
            # Disable automatic batching as we handle batching ourselves
            "batch_size": None,
            "collate_fn": None,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": seed_worker,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        return DataLoader(train_dataset, **dataloader_params)  # type: ignore

    def get_eval_dataloader(self, eval_dataset: str | Dataset | None = None) -> DataLoader:
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = cast("Chronos2Dataset", self.eval_dataset)

        if self.args.eval_batch_size > eval_dataset.batch_size:
            warnings.warn(
                f"The batch_size of the eval_dataset ({eval_dataset.batch_size}) does not match the batch_size "
                f"in TrainingArguments ({self.args.eval_batch_size}). On machines with multiple GPUs, this may indicate "
                f"that multiple GPUs are visible and transformers is using DataParallel for training by default. "
                f"This may lead to unnecessary slowdown and unexpected behavior. We strongly recommend setting the CUDA_VISIBLE_DEVICES "
                f"environment variable to ensure that only a single GPU is visible.",
                category=UserWarning,
                stacklevel=3,
            )

        dataloader_params = {
            # Disable automatic batching as we handle batching ourselves
            "batch_size": None,
            "collate_fn": None,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
            "prefetch_factor": self.args.dataloader_prefetch_factor,
        }

        return DataLoader(eval_dataset, **dataloader_params)  # type: ignore


import torch

import torch

def check_tensor_nan(obj, name="tensor"):
    """
    递归检查 dict / list / tuple / tensor 中是否有 NaN / Inf
    发现直接报错，避免 DDP 卡死
    """
    if torch.is_tensor(obj):
        if not torch.isfinite(obj).all():
            nan_mask = torch.isnan(obj)
            inf_mask = torch.isinf(obj)
            print(f"❌ {name} 有NaN/Inf:")
            print(f"  Inf位置: {torch.where(inf_mask)}")
            print("行 169:", obj[169, 465:475])  # 查看附近列
            print("列 470:", obj[165:175, 470])  # 查看附近行

            raise RuntimeError(f"NaN/Inf detected in {name}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            check_tensor_nan(v, f"{name}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            check_tensor_nan(v, f"{name}[{i}]")


class DebugGradTrainer(Chronos2Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()

        # inputs 已经是一个完整 batch（Chronos2Dataset 保证）
        inputs = self._prepare_inputs(inputs)
        check_tensor_nan(inputs["context"], "inputs")
        
        loss = self.compute_loss(model, inputs)

        # 只允许检查 loss（不会引入同步死锁）
        if not torch.isfinite(loss):
            raise RuntimeError("Loss is NaN/Inf")
        
        

        self.accelerator.backward(loss)

        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print(f"❌ NaN grad in {name}")
                raise RuntimeError("NaN grad detected")

        return loss.detach()


