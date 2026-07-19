import os
import glob
import math
import logging
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset

from chronos.base import BaseChronosPipeline, ForecastType
from chronos.chronos2 import Chronos2Model
from chronos.chronos2.dataset import Chronos2Dataset, DatasetMode, TensorOrArray

from transformers import TrainerCallback

if TYPE_CHECKING:
    import datasets
    import fev
    import pandas as pd
    from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


class MixedMultiRootDataset(IterableDataset):
    """
    Iterate over multiple Chronos2Dataset instances without merging physical files.

    When shuffle=True, one cycle still consumes approximately one pass over all
    child datasets, but the batch source order is shuffled, e.g. A, B, B, A.
    This avoids the undesirable behavior of finishing all batches from A before B,
    while keeping the same data files and without copying .npy files.

    When shuffle=False, it falls back to sequential chaining. This is useful for
    validation, where deterministic evaluation is usually preferable.
    """

    def __init__(
        self,
        datasets: Sequence[IterableDataset],
        batches_per_dataset: Sequence[int],
        repeat: bool,
        shuffle: bool,
        batch_size: int,
    ):
        super().__init__()
        self.datasets = list(datasets)
        self.batches_per_dataset = [int(x) for x in batches_per_dataset]
        self.repeat = repeat
        self.shuffle = shuffle
        self.batch_size = int(batch_size)

        if len(self.datasets) != len(self.batches_per_dataset):
            raise ValueError("datasets 和 batches_per_dataset 长度不一致。")
        if len(self.datasets) == 0:
            raise ValueError("MixedMultiRootDataset 至少需要一个子数据集。")
        if sum(max(0, x) for x in self.batches_per_dataset) <= 0:
            raise ValueError("batches_per_dataset 全部为 0，无法训练。")

    def __iter__(self):
        # DataLoader num_workers=0 in your TrainingArguments, so one iterator is enough here.
        rng = np.random.default_rng()

        while True:
            iterators = [iter(dataset) for dataset in self.datasets]

            order = []
            for dataset_idx, max_batches in enumerate(self.batches_per_dataset):
                if max_batches > 0:
                    order.extend([dataset_idx] * max_batches)

            if self.shuffle:
                rng.shuffle(order)

            yielded_any = False
            for dataset_idx in order:
                try:
                    item = next(iterators[dataset_idx])
                except StopIteration:
                    # If one child dataset is unexpectedly shorter than estimated,
                    # skip it instead of resetting it within the same cycle.
                    continue
                yielded_any = True
                yield item

            if not self.repeat or not yielded_any:
                break



def _is_multi_datapath(datapath) -> bool:
    return isinstance(datapath, (list, tuple))


def _inputs_embed_data_root(inputs) -> bool:
    return bool(inputs) and isinstance(inputs[0], (list, tuple)) and len(inputs[0]) >= 7


def _count_index_rows(datapath: str, flag: str) -> int:
    """Count rows in datapath/DataIndex/{flag}/*.csv without changing train.py."""
    import pandas as pd

    files = sorted(glob.glob(f"{datapath}/DataIndex/{flag}/*.csv"))
    total = 0
    for file in files:
        total += len(pd.read_csv(file).values.tolist())
    return total


def _split_inputs_by_datapath(datapaths: Sequence[str], inputs, flag: str):
    """
    Split the flattened inputs list back into per-root inputs.

    This matches the current train.py behavior: InputConvert([A, B], flag) appends
    all A index rows first, then all B index rows. We recover the split sizes by
    reading DataIndex/{flag} under each root.
    """
    if inputs is None:
        return []

    inputs = list(inputs)
    counts = [_count_index_rows(str(one_datapath), flag) for one_datapath in datapaths]
    expected_total = sum(counts)

    if expected_total != len(inputs):
        raise ValueError(
            f"无法按数据根目录切分 {flag}_inputs："
            f"DataIndex 统计总数={expected_total}, 实际传入 inputs 数={len(inputs)}。"
            f"请确认 train.py 的 InputConvert 顺序仍然是按 datapath 列表依次拼接，"
            f"并且 DataIndex/{flag}/*.csv 没有在读取后发生变化。"
        )

    result = []
    start = 0
    for count in counts:
        result.append(inputs[start:start + count])
        start += count
    return result


def _make_chronos2_dataset(
    datapath,
    inputs,
    flag: str,
    context_length: int,
    prediction_length: int,
    batch_size: int,
    output_patch_size: int,
    mode: DatasetMode,
    min_past: int | None = None,
):
    """
    Create a single-root or multi-root Chronos2Dataset.

    Single-root: exactly the same as the original logic.
    Multi-root: construct one Chronos2Dataset per root and chain them sequentially.
    """
    if not _is_multi_datapath(datapath):
        kwargs = dict(
            datapath=datapath,
            inputs=inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=output_patch_size,
            mode=mode,
        )
        if min_past is not None:
            kwargs["min_past"] = min_past
        return Chronos2Dataset(**kwargs)

    if _inputs_embed_data_root(inputs):
        kwargs = dict(
            datapath=str(datapath[0]),
            inputs=inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=output_patch_size,
            mode=mode,
        )
        if min_past is not None:
            kwargs["min_past"] = min_past
        return Chronos2Dataset(**kwargs)

    datapaths = [str(one_datapath) for one_datapath in datapath]
    split_inputs = _split_inputs_by_datapath(datapaths, inputs, flag)

    datasets = []
    batches_per_dataset = []
    for one_datapath, one_inputs in zip(datapaths, split_inputs):
        if len(one_inputs) == 0:
            continue

        kwargs = dict(
            datapath=one_datapath,
            inputs=one_inputs,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=output_patch_size,
            mode=mode,
        )
        if min_past is not None:
            kwargs["min_past"] = min_past

        datasets.append(Chronos2Dataset(**kwargs))
        batches_per_dataset.append(max(1, math.ceil(len(one_inputs) / batch_size)))

    if len(datasets) == 0:
        return None

    return MixedMultiRootDataset(
        datasets=datasets,
        batches_per_dataset=batches_per_dataset,
        repeat=(mode == DatasetMode.TRAIN),
        shuffle=(mode == DatasetMode.TRAIN),
        batch_size=batch_size,
    )


class Train(nn.Module):
    def __init__(self, model: Chronos2Model):
        super().__init__()
        self.model = model

    @property
    def model_context_length(self) -> int:
        return self.model.module.chronos_config.context_length

    @property
    def model_output_patch_size(self) -> int:
        return self.model.module.chronos_config.output_patch_size

    @property
    def model_prediction_length(self) -> int:
        return self.model.module.chronos_config.max_output_patches * self.model.module.chronos_config.output_patch_size

    @property
    def quantiles(self) -> list[float]:
        return self.model.module.chronos_config.quantiles

    @property
    def max_output_patches(self) -> int:
        return self.model.module.chronos_config.max_output_patches

    def fit(
        self,
        datapath: str | Sequence[str] | None = None,
        inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]]
        | None = None,
        prediction_length: int | None = None,
        validation_inputs: TensorOrArray
        | Sequence[TensorOrArray]
        | Sequence[Mapping[str, TensorOrArray | Mapping[str, TensorOrArray | None]]]
        | None = None,
        context_length: int | None = None,
        learning_rate: float = 1e-4,
        num_steps: int = 1,
        batch_size: int = 256,
        gradient_accumulation_steps: int = 1,
        save_steps: int = 1000,
        eval_steps: int = 1000,
        output_dir: Path | str | None = None,
        min_past: int | None = None,
        callbacks: list["TrainerCallback"] | None = None,
        resume_checkpoint: str | None = None,
        **extra_trainer_kwargs
    ):
        """
        Parameters
        ----------
        inputs
            The time series on which the model will be trained. The allowed formats of inputs are the same as `Chronos2Pipeline.predict()`.
        prediction_length
            The prediction horizon for which the model will be trained
        validation_inputs
            The time series used for validation and model selection. The format of `validation_inputs` is exactly the same as `inputs`, by default None which
            means that no validation is performed.
        context_length
            The maximum context length used during training, by default set to the model's default context length
        learning_rate
            The learning rate for the optimizer, by default 1e-4
        num_steps
            The number of steps to train for, by default 1
        batch_size
            The batch size used for training. Note that the batch size here means the number of time series, including target(s) and covariates,
            which are input into the model. If your data has multiple target and/or covariates, the effective number of time series tasks in a batch
            will be lower than this value, by default 256
        output_dir
            The directory in which outputs from the `Trainer` will be saved
        min_past
            The minimum number of time steps the context must have during training. All time series shorter than `min_past + prediction_length`
            are filtered out, by default set equal to prediction_length
        callbacks
            A list of `TrainerCallback`s which will be forwarded to the HuggingFace `Trainer`
        **extra_trainer_kwargs
            Extra kwargs are directly forwarded to `TrainingArguments`

        Returns
        -------
        A pretrained model
        """

        import torch.cuda
        from transformers.training_args import TrainingArguments

        from chronos.chronos2.trainer import Chronos2Trainer, EvaluateAndSaveFinalStepCallback

        model = self.model# type: ignore

        # 兼容外层训练脚本传入 data_paths=... 的写法，同时避免该参数继续传给 TrainingArguments。
        if datapath is None:
            datapath = extra_trainer_kwargs.pop("data_paths", None)
        else:
            extra_trainer_kwargs.pop("data_paths", None)

        if datapath is None:
            raise ValueError("datapath is required. Please pass datapath=... or data_paths=... to Train.fit().")
        if inputs is None:
            raise ValueError("inputs is required.")
        if prediction_length is None:
            raise ValueError("prediction_length is required.")

        if context_length is None:
            context_length = self.model_context_length

        if min_past is None:
            min_past = prediction_length

        train_dataset = _make_chronos2_dataset(
            datapath=datapath,
            inputs=inputs,
            flag="train",
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            output_patch_size=self.model_output_patch_size,
            min_past=min_past,
            mode=DatasetMode.TRAIN,
        )
        if train_dataset is None:
            raise ValueError("train_dataset 为空，请检查 DataIndex/train/*.csv。")
        """
        j=0
        for i,data in enumerate(train_dataset):
            j=j+1
            if j %100==0:
                print(j)
        """

        #for i in train_dataset:
        #    print(i["context"].shape)

        if output_dir is None:
            output_dir = Path("chronos-2-pretrain") / time.strftime("%Y-%m-%d_%H-%M-%S")
        elif isinstance(output_dir, str):
            output_dir = Path(output_dir)

        assert isinstance(output_dir, Path)

        use_cpu = str(self.model.device) == "cpu"
        has_sm80 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

        # warn user if a cuda device is available and CPU fine-tuning is used
        if use_cpu and torch.cuda.is_available():
            warnings.warn(
                "The model is being fine-tuned on the CPU, but a CUDA device is available. "
                "We recommend using the GPU for faster fine-tuning.",
                category=UserWarning,
                stacklevel=2,
            )
        local_rank=int(os.environ.get('LOCAL_RANK', -1))
        training_kwargs: dict = dict(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            warmup_ratio=0.0,
            optim="adamw_torch_fused",
            logging_strategy="steps",
            logging_steps=10,
            disable_tqdm=False,
            report_to="none",
            max_steps=num_steps,
            max_grad_norm=1.0,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=0,
            tf32=has_sm80 and not use_cpu,
            #bf16=has_sm80 and not use_cpu,
            save_only_model=False,
            prediction_loss_only=True,
            save_strategy="steps",
            save_steps=1000,
            eval_strategy="no",
            eval_steps=None,
            load_best_model_at_end=False,
            metric_for_best_model=None,
            use_cpu=use_cpu,
            local_rank=local_rank
        )

        eval_dataset = None
        callbacks = callbacks or []
        if validation_inputs is not None and not (isinstance(validation_inputs, Sequence) and len(validation_inputs) == 0):
            # construct validation dataset
            eval_dataset = _make_chronos2_dataset(
                datapath=datapath,
                inputs=validation_inputs,
                flag="val",
                context_length=context_length,
                prediction_length=prediction_length,
                batch_size=batch_size,
                output_patch_size=self.model_output_patch_size,
                mode=DatasetMode.VALIDATION,
            )

            if eval_dataset is not None:
                # set validation parameters
                training_kwargs["eval_strategy"] = "steps"
                training_kwargs["eval_steps"] = eval_steps
                training_kwargs["save_strategy"] = "steps"
                training_kwargs["save_steps"] = save_steps
                training_kwargs["load_best_model_at_end"] = True
                training_kwargs["metric_for_best_model"] = "eval_loss"
                training_kwargs["label_names"] = ["future_target"]

                # add callback to ensure that the final model is evaluated
                callbacks.append(EvaluateAndSaveFinalStepCallback())
        training_kwargs.update(extra_trainer_kwargs)

        if training_kwargs["tf32"]:
            # setting tf32=True changes these global properties, we copy them here so that
            # we can restore them after fine-tuning
            matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
            cudnn_tf32 = torch.backends.cudnn.allow_tf32

        training_args = TrainingArguments(**training_kwargs)

        callbacks = callbacks or []
        trainer = Chronos2Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
        )

        #if remove_printer_callback:
            #trainer.pop_callback(PrinterCallback)
        if resume_checkpoint:
            print(resume_checkpoint)
            print(output_dir)
            trainer.train(resume_from_checkpoint=True)
        else:
            trainer.train()


        if training_kwargs["tf32"]:
            # restore tf32 settings
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
        
        return model
