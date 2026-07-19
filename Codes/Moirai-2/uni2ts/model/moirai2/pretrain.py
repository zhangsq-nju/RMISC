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

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any, Optional

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import nn

from uni2ts.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedPointLoss,
    PackedQuantileMAELoss,
)
from uni2ts.module.norm import RMSNorm
from uni2ts.module.position import (
    BinaryAttentionBias,
    LearnedEmbedding,
    LearnedProjection,
)
from uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear
from uni2ts.optim import SchedulerType, get_scheduler
from uni2ts.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    DummyValueImputation,
    FlatPackCollection,
    FlatPackFields,
    ImputeTimeSeries,
    LambdaSetFieldIfNotPresent,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SelectFields,
    SetValue,
    SequencifyField,
    Transformation,
)

from .module import Moirai2Module


class Moirai2Pretrain(L.LightningModule):
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    )
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "patch_size": np.zeros,
    }

    def __init__(
        self,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        max_dim: int,
        num_training_steps: int,
        num_warmup_steps: int,
        module_kwargs: Optional[dict[str, Any]] = None,
        module: Optional[Moirai2Module] = None,
        num_samples: int = 100,
        beta1: float = 0.9,
        beta2: float = 0.98,
        loss_func: PackedLoss = PackedQuantileMAELoss(),
        val_metric: Optional[PackedLoss | list[PackedLoss]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        log_on_step: bool = False,
    ):
        assert (module is not None) or (
            module_kwargs is not None
        ), "if module is not provided, module_kwargs is required"
        assert (
            num_warmup_steps <= num_training_steps
        ), f"num_warmup_steps ({num_warmup_steps}) should be <= num_training_steps ({num_training_steps})."
        super().__init__()
        self.save_hyperparameters(ignore=["module"])
        self.module = Moirai2Module(**module_kwargs) if module is None else module

    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
        training_mode: Bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Redirects to the forward function of MoiraiModule.

        :param target: input data
        :param observed_mask: binary mask for missing values, 1 if observed, 0 otherwise
        :param sample_id: indices indicating the sample index (for packing)
        :param time_id: indices indicating the time index
        :param variate_id: indices indicating the variate index
        :param prediction_mask: binary mask for prediction horizon, 1 if part of the horizon, 0 otherwise
        :param patch_size: patch size for each token
        :param training_mode: Moirai2Module training/eval output mode
        :return: preds and scaled_target when training_mode=True
        """
        ### 修改了Moirai2Pretrain.forward：统一调用module.forward。
        outputs = self.module(
            target=target,
            observed_mask=observed_mask,
            sample_id=sample_id,
            time_id=time_id,
            variate_id=variate_id,
            prediction_mask=prediction_mask,
            training_mode=training_mode,
        )
        return outputs

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Implements LightningModule training_step. Logs training loss.

        :param batch: batched inputs
        :param batch_idx: index of current batch
        :return: training loss for current batch
        """
        preds, scaled_target = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]},
            training_mode=True,
        )
        ### 修改了训练loss输入：最后一个context token一次对齐整个forecast horizon。
        (
            decoder_preds,
            decoder_target,
            decoder_prediction_mask,
            decoder_observed_mask,
            decoder_sample_id,
            decoder_variate_id,
        ) = self._build_decoder_only_loss_inputs(
            preds=preds,
            scaled_target=scaled_target,
            observed_mask=batch["observed_mask"],
            prediction_mask=batch["prediction_mask"],
            sample_id=batch["sample_id"],
            time_id=batch["time_id"],
            variate_id=batch["variate_id"],
        )
        loss = self.hparams.loss_func(
            pred=decoder_preds,
            **{
                "target": decoder_target,
                "sample_id": decoder_sample_id,
                "variate_id": decoder_variate_id,
                "prediction_mask": decoder_prediction_mask,
                "observed_mask": decoder_observed_mask,
            },
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"train/{self.hparams.loss_func.__class__.__name__}",
            loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """
        Implements LightningModule validation_step. Logs validation loss and additional metrics from val_metric.

        :param batch:
        :param batch_idx:
        :param dataloader_idx:
        :return: validation loss for current batch
        """
        preds, scaled_target = self(
            **{field: batch[field] for field in list(self.seq_fields) + ["sample_id"]},
            training_mode=True,  ### 修改了验证输出：按训练模式返回preds和scaled_target计算loss。
        )
        (
            decoder_preds,
            decoder_target,
            decoder_prediction_mask,
            decoder_observed_mask,
            decoder_sample_id,
            decoder_variate_id,
        ) = self._build_decoder_only_loss_inputs(
            preds=preds,
            scaled_target=scaled_target,
            observed_mask=batch["observed_mask"],
            prediction_mask=batch["prediction_mask"],
            sample_id=batch["sample_id"],
            time_id=batch["time_id"],
            variate_id=batch["variate_id"],
        )
        ### 修改了验证loss计算：使用forecast horizon对齐后的target和mask。
        val_loss = self.hparams.loss_func(
            pred=decoder_preds,
            **{
                "target": decoder_target,
                "sample_id": decoder_sample_id,
                "variate_id": decoder_variate_id,
                "prediction_mask": decoder_prediction_mask,
                "observed_mask": decoder_observed_mask,
            },
        )
        batch_size = (
            batch["sample_id"].max(dim=1).values.sum() if "sample_id" in batch else None
        )
        self.log(
            f"val/{self.hparams.loss_func.__class__.__name__}",
            val_loss,
            on_step=self.hparams.log_on_step,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )

        if self.hparams.val_metric is not None:
            val_metrics = (
                self.hparams.val_metric
                if isinstance(self.hparams.val_metric, list)
                else [self.hparams.val_metric]
            )
            for metric_func in val_metrics:
                if isinstance(metric_func, PackedPointLoss):
                    # 淇敼浜嗕粠quantile杈撳嚭鍙栦腑浣嶆暟鐨勬柟娉?                    median_idx = self.module.quantile_levels.index(0.5)
                    pred = decoder_preds.reshape(
                        *decoder_preds.shape[:-1],
                        self.module.num_quantiles,
                        self.module.patch_size,
                    )[..., median_idx, :]
                elif isinstance(metric_func, PackedDistributionLoss):
                    raise ValueError(
                        "PackedDistributionLoss is not supported by Moirai2Pretrain."
                    )
                else:
                    pred = decoder_preds  # Moirai2鐨剄uantile杈撳嚭

                metric = metric_func(
                    pred=pred,
                    **{
                        "target": decoder_target,
                        "sample_id": decoder_sample_id,
                        "variate_id": decoder_variate_id,
                        "prediction_mask": decoder_prediction_mask,
                        "observed_mask": decoder_observed_mask,
                    },
                )

                self.log(
                    f"val/{metric_func.__class__.__name__}",
                    metric,
                    on_step=self.hparams.log_on_step,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    rank_zero_only=True,
                )

        return val_loss

    def _build_decoder_only_loss_inputs(
        self,
        preds: torch.Tensor,
        scaled_target: torch.Tensor,
        observed_mask: torch.Tensor,
        prediction_mask: torch.Tensor,
        sample_id: torch.Tensor,
        time_id: torch.Tensor,
        variate_id: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        ### 修改了Moirai2预训练loss对齐：只用prediction窗口前最后一个context token匹配整个forecast horizon的loss。但真正的预测使用前面所有的context token预测所有的个prediction token
        num_predict_token = self.module.num_predict_token
        if num_predict_token <= 0:
            raise ValueError(f"num_predict_token must be positive, got {num_predict_token}.")

        pred_dim = preds.shape[-1]
        if pred_dim % num_predict_token != 0:
            raise ValueError(
                f"Prediction dim {pred_dim} is not divisible by "
                f"num_predict_token={num_predict_token}."
            )
        per_token_pred_dim = pred_dim // num_predict_token
        preds = preds.reshape(*preds.shape[:-1], num_predict_token, per_token_pred_dim)

        source_sample_id = sample_id.unsqueeze(-1)
        target_sample_id = sample_id.unsqueeze(-2)
        source_variate_id = variate_id.unsqueeze(-1)
        target_variate_id = variate_id.unsqueeze(-2)
        same_series = torch.eq(source_sample_id, target_sample_id) & torch.eq(
            source_variate_id, target_variate_id
        )

        target_time_for_min = time_id.unsqueeze(-2).expand_as(same_series)
        large_time = torch.full_like(target_time_for_min, torch.iinfo(time_id.dtype).max)
        first_prediction_time = torch.where(
            same_series & prediction_mask.unsqueeze(-2),
            target_time_for_min,
            large_time,
        ).amin(dim=-1)
        has_prediction_target = first_prediction_time.ne(torch.iinfo(time_id.dtype).max)
        is_last_context_token = (
            sample_id.ne(0)
            & ~prediction_mask
            & has_prediction_target
            & torch.eq(time_id, first_prediction_time - 1)
        )

        offsets = torch.arange(
            1,
            num_predict_token + 1,
            device=time_id.device,
            dtype=time_id.dtype,
        )
        offset_shape = (1,) * (time_id.ndim - 1) + (1, num_predict_token, 1)
        offsets = offsets.reshape(offset_shape)
        source_time_id = time_id.unsqueeze(-1).unsqueeze(-1)
        target_time_id = time_id.unsqueeze(-2).unsqueeze(-2)

        forecast_mask = (
            same_series.unsqueeze(-2)
            & is_last_context_token.unsqueeze(-1).unsqueeze(-1)
            & prediction_mask.unsqueeze(-2).unsqueeze(-2)
            & torch.eq(target_time_id, source_time_id + offsets)
        )

        match_weight = forecast_mask.to(dtype=scaled_target.dtype)
        decoder_target = torch.einsum("...skt,...tp->...skp", match_weight, scaled_target)
        decoder_observed_mask = (
            torch.einsum(
                "...skt,...tp->...skp",
                match_weight,
                observed_mask.to(dtype=scaled_target.dtype),
            )
            > 0
        )
        decoder_prediction_mask = forecast_mask.any(dim=-1)
        decoder_sample_id = sample_id.unsqueeze(-1).expand_as(decoder_prediction_mask)
        decoder_variate_id = variate_id.unsqueeze(-1).expand_as(decoder_prediction_mask)

        flat_shape = (*preds.shape[:-3], -1)
        decoder_preds = preds.reshape(*flat_shape, per_token_pred_dim)
        decoder_target = decoder_target.reshape(*flat_shape, decoder_target.shape[-1])
        decoder_observed_mask = decoder_observed_mask.reshape(
            *flat_shape, decoder_observed_mask.shape[-1]
        )
        decoder_prediction_mask = decoder_prediction_mask.reshape(*flat_shape)
        decoder_sample_id = decoder_sample_id.reshape(*flat_shape)
        decoder_variate_id = decoder_variate_id.reshape(*flat_shape)

        return (
            decoder_preds,
            decoder_target,
            decoder_prediction_mask,
            decoder_observed_mask,
            decoder_sample_id,
            decoder_variate_id,
        )

    def configure_optimizers(self) -> dict:
        """
        Implements LightningModule configure_optimizers which defines the configuration of optimizer and learning rate
        scheduler.

        :return: dictionary of optimizers and learning rate schedulers
        """
        decay = set()
        no_decay = set()

        whitelist_params = (
            LearnedProjection,
            MultiInSizeLinear,
            MultiOutSizeLinear,
            nn.Linear,
        )
        blacklist_params = (
            BinaryAttentionBias,
            LearnedEmbedding,
            RMSNorm,
            nn.Embedding,
            nn.LayerNorm,
        )

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_params):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_params):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

        optim_groups = [
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(decay))],
                ),
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [param_dict[pn] for pn in sorted(list(no_decay))],
                ),
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=1e-6,
        )
        scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer,
            num_warmup_steps=self.hparams.num_warmup_steps,
            num_training_steps=self.hparams.num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "step",
            },
        }

    @property
    def train_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        """
        Get a dictionary of Transforms, with a default Transform as defined:
        SampleDimension: Subsample the variate dimension of a time series
        LambdaSetFieldIfNotPresent: Use the module patch size as the fixed patch size.
        PatchCrop: Perform cropping on the time series
        PackFields: Pack target only.
        ImputeTimeSeries: Imputes missing values with 0
        Patchify: Perform patching
        AddVariateIndex: Add variate_id feature
        AddTimeIndex: Add time_id feature
        MaskedPrediction: Specify the task,
            i.e., sample the total input length, as well as sample the proportion of look-back window and prediction window length.
        FlatPackCollection: Pack/Merge along 'variate_id, time_id, prediction_mask, observed_mask, and target' dimensions.
        FlatPackFields: Pack/Merge 'target'.
        SequencifyField: sequencify the 'patch_size' field.
        SelectFields: Output the data of predefined fields

        :return: defaultdict with default Transform
        """

        def default_train_transform(prediction_length: int | None = None):
            ### 修改了Moirai2训练transform：删除cov/feat_dynamic_real链路，只保留单维target。
            return (
                LambdaSetFieldIfNotPresent(
                    field="patch_size",
                    get_value=SetValue(np.int64(self.module.patch_size)),
                )
                + PatchCrop(
                    max_patches=self.module.max_seq_len,
                    will_flatten=False,  ### 修改了PatchCrop：单维target不再按多变量flatten预算长度。
                    fields=("target",),
                    optional_fields=tuple(),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                    feat=False,
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=tuple(),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=tuple(),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=self.module.patch_size,
                    fields=("target", "observed_mask"),
                    optional_fields=tuple(),
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=tuple(),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=1,  ### 修改了variate_id范围：单维输入只允许变量0。
                    randomize=False,  ### 修改了variate_id随机化：稳定生成0。
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=tuple(),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + MaskedPrediction(
                    prediction_length=prediction_length,
                    target_field="target",
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + FlatPackCollection(field="variate_id", feat=False)
                + FlatPackCollection(field="time_id", feat=False)
                + FlatPackCollection(field="prediction_mask", feat=False)
                + FlatPackCollection(field="observed_mask", feat=True)
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=tuple(),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )

        return defaultdict(lambda: default_train_transform)
    



    @property
    def val_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        """
        Get a dictionary of Transforms, with a default Transform as defined:
        SampleDimension: Subsample the variate dimension of a time series
        LambdaSetFieldIfNotPresent: Use the module patch size as the fixed patch size.
        PatchCrop: Perform cropping on the time series
        PackFields: Pack target only.
        ImputeTimeSeries: Imputes missing values with 0
        Patchify: Perform patching
        AddVariateIndex: Add variate_id feature
        AddTimeIndex: Add time_id feature
        MaskedPrediction: Specify the task,
            i.e., sample the total input length, as well as sample the proportion of look-back window and prediction window length.
        FlatPackCollection: Pack/Merge along 'variate_id, time_id, prediction_mask, observed_mask, and target' dimensions.
        FlatPackFields: Pack/Merge 'target'.
        SequencifyField: sequencify the 'patch_size' field.
        SelectFields: Output the data of predefined fields

        :return: defaultdict with default Transform
        """

        def default_val_transform(prediction_length: int | None = None):
            ### 修改了Moirai2验证transform：删除cov/feat_dynamic_real链路，只保留单维target。
            return (
                LambdaSetFieldIfNotPresent(
                    field="patch_size",
                    get_value=SetValue(np.int64(self.module.patch_size)),
                )
                + PatchCrop(
                    max_patches=self.module.max_seq_len,
                    will_flatten=False,  ### 修改了PatchCrop：单维target不再按多变量flatten预算长度。
                    fields=("target",),
                    optional_fields=tuple(),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                    feat=False,
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=tuple(),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=tuple(),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=self.module.patch_size,
                    fields=("target", "observed_mask"),
                    optional_fields=tuple(),
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=tuple(),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=1,  ### 修改了variate_id范围：单维输入只允许变量0。
                    randomize=False,  ### 修改了variate_id随机化：稳定生成0。
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=tuple(),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + MaskedPrediction(
                    prediction_length=prediction_length,
                    target_field="target",
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + FlatPackCollection(field="variate_id", feat=False)
                + FlatPackCollection(field="time_id", feat=False)
                + FlatPackCollection(field="prediction_mask", feat=False)
                + FlatPackCollection(field="observed_mask", feat=True)
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=tuple(),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )

        return defaultdict(lambda: default_val_transform)

