import argparse
import datetime
import json
import logging
import math
import os
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_TRANSFORMERS = PROJECT_DIR / "official_sources" / "transformers" / "src"
if LOCAL_TRANSFORMERS.is_dir():
    sys.path.insert(0, str(LOCAL_TRANSFORMERS))

import numpy as np
import torch
from transformers import TrainingArguments
from transformers.models.timesfm2_5.configuration_timesfm2_5 import TimesFm2_5Config
from transformers.models.timesfm2_5.modeling_timesfm2_5 import TimesFm2_5ModelForPrediction

from timesfm_train.data import (
    count_target_series,
    filter_rows,
    input_convert,
    make_dataset,
    resolve_data_paths,
)
from timesfm_train.trainer import EvaluateAndSaveFinalStepCallback, TimesFMTrainer


os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "10000"
logger = logging.getLogger(__file__)


class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


def is_main_process() -> bool:
    return int(os.environ.get("RANK", 0)) == 0


def log_on_main(msg: str, log_level: int = logging.INFO):
    if is_main_process():
        logger.log(log_level, msg)


def setup_output_capture(output_dir: Path):
    if not is_main_process():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    log_fp = open(output_dir / "train.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_fp)
    sys.stderr = _TeeStream(sys.stderr, log_fp)
    return log_fp


def get_next_path(
    base_fname: str,
    base_dir: Path,
    file_type: str = "",
    separator: str = "-",
) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    if file_type == "":
        items = filter(
            lambda x: x.is_dir() and re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob("*"),
        )
    else:
        items = filter(
            lambda x: re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob(f"*.{file_type}"),
        )
    run_nums = [int(x.stem.replace(base_fname + separator, "")) for x in items] + [-1]
    fname = f"{base_fname}{separator}{max(run_nums) + 1}" + (f".{file_type}" if file_type else "")
    return base_dir / fname


def get_shared_output_dir(base_dir: Path) -> Path:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return get_next_path("run", base_dir=base_dir, file_type="")

    output_dir_holder = [
        str(get_next_path("run", base_dir=base_dir, file_type="")) if is_main_process() else None
    ]
    torch.distributed.broadcast_object_list(output_dir_holder, src=0)
    return Path(output_dir_holder[0])


def init_distributed_if_needed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return

    global_rank = int(os.environ.get("RANK", 0))
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        local_rank = int(os.environ.get("LOCAL_RANK", global_rank % max(1, num_gpus)))
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        world_size=world_size,
        rank=global_rank,
        timeout=datetime.timedelta(seconds=30 * 60),
    )


def get_training_job_info() -> Dict:
    job_info: Dict = {
        "python_version": sys.version.replace("\n", " "),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        job_info["device_count"] = torch.cuda.device_count()
        job_info["device_names"] = {
            idx: torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())
        }
        job_info["mem_info"] = {
            idx: torch.cuda.mem_get_info(device=idx) for idx in range(torch.cuda.device_count())
        }
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        job_info["world_size"] = torch.distributed.get_world_size()
    return job_info


def save_training_info(ckpt_path: Path, training_config: Dict):
    ckpt_path.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path / "training_info.json", "w", encoding="utf-8") as fp:
        json.dump(
            {"training_config": training_config, "job_info": get_training_job_info()},
            fp,
            indent=4,
            ensure_ascii=False,
        )


def load_model_config(config_path: Path) -> tuple[TimesFm2_5Config, Dict]:
    with open(config_path, "r", encoding="utf-8") as fp:
        raw_config = json.load(fp)
    config = TimesFm2_5Config(**raw_config)
    return config, raw_config


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TimesFM-2.5 from scratch on indexed datasets.")
    parser.add_argument("--training-data-paths", default="data/large")
    parser.add_argument("--config-path", default=str(PROJECT_DIR / "config.json"))
    parser.add_argument("--context-length", type=int, default=1984)
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--min-past", type=int, default=64)
    parser.add_argument("--per-device-train-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--num-steps", type=int, default=200000)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=2000)
    parser.add_argument("--point-loss", choices=["mse", "huber"], default="mse")
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--quantile-loss-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "output"))
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    return parser


def main():
    args = build_arg_parser().parse_args()

    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.setLevel(logging.INFO)
    init_distributed_if_needed()

    data_paths_list = resolve_data_paths(args.training_data_paths)
    data_paths = data_paths_list[0] if len(data_paths_list) == 1 else data_paths_list

    output_base_dir = Path(args.output_dir)
    output_dir = Path(args.resume_checkpoint).resolve().parent if args.resume_checkpoint else get_shared_output_dir(output_base_dir)
    setup_output_capture(output_dir)

    log_on_main(f"Logging dir: {output_dir}")
    log_on_main(f"Resolved training data roots: {data_paths_list}")
    log_on_main("Loading DataIndex rows")

    train_inputs = input_convert(data_paths, "train")
    val_inputs = input_convert(data_paths, "val")
    train_inputs_filtered = filter_rows(train_inputs, args.min_past + args.prediction_length)
    val_inputs_filtered = filter_rows(val_inputs, args.min_past + args.prediction_length)

    train_target_series = count_target_series(train_inputs_filtered)
    val_target_series = count_target_series(val_inputs_filtered)
    if train_target_series <= 0:
        raise ValueError("train_inputs 为空，无法训练。请检查 DataIndex/train/*.csv 和 target 字段。")

    log_on_main(
        f"Loaded train rows={len(train_inputs)}, filtered rows={len(train_inputs_filtered)}, "
        f"target_series={train_target_series}; val rows={len(val_inputs)}, "
        f"filtered rows={len(val_inputs_filtered)}, target_series={val_target_series}"
    )

    config, raw_config = load_model_config(Path(args.config_path))
    config.context_length = args.context_length
    config.horizon_length = args.prediction_length
    config.force_flip_invariance = False
    config.infer_is_positive = False
    if args.context_length % config.patch_length != 0:
        raise ValueError(
            f"context_length 必须能被 patch_length 整除: {args.context_length} % {config.patch_length} != 0"
        )

    log_on_main("Initializing TimesFM-2.5 from scratch")
    model = TimesFm2_5ModelForPrediction(config)
    log_on_main(
        f"Loss setup: point_loss={args.point_loss}, huber_beta={args.huber_beta}, "
        f"quantile_loss_weight={args.quantile_loss_weight}"
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    num_steps = args.num_steps
    save_steps = args.save_steps
    if args.num_epochs > 0:
        effective_samples_per_step = (
            args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps
        )
        steps_per_epoch = math.ceil(train_target_series / effective_samples_per_step)
        num_steps = max(1, steps_per_epoch * args.num_epochs)
        log_on_main(
            f"Using epoch-based stopping: num_epochs={args.num_epochs}, "
            f"train_target_series={train_target_series}, world_size={world_size}, "
            f"per_device_train_batch_size={args.per_device_train_batch_size}, "
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "
            f"effective_samples_per_step={effective_samples_per_step}, "
            f"steps_per_epoch={steps_per_epoch}, num_steps={num_steps}"
        )
    else:
        log_on_main(f"Using step-based stopping: num_steps={num_steps}")

    mid_eval_step = max(1, math.ceil(num_steps / 2))
    eval_steps = mid_eval_step
    save_steps = mid_eval_step if save_steps <= 0 else min(save_steps, mid_eval_step)
    log_on_main(
        f"Evaluation schedule: at step {mid_eval_step} and final step {num_steps}; "
        f"save_steps={save_steps}"
    )

    train_dataset = make_dataset(
        data_paths=data_paths,
        rows=train_inputs_filtered,
        flag="train",
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        batch_size=args.per_device_train_batch_size,
        min_past=args.min_past,
        mode="train",
    )

    eval_dataset = None
    callbacks = []
    eval_strategy = "no"
    if val_target_series > 0:
        eval_dataset = make_dataset(
            data_paths=data_paths,
            rows=val_inputs_filtered,
            flag="val",
            context_length=args.context_length,
            prediction_length=args.prediction_length,
            batch_size=args.per_device_train_batch_size,
            min_past=args.min_past,
            mode="val",
        )
        eval_strategy = "steps"
        callbacks.append(EvaluateAndSaveFinalStepCallback())

    has_sm80 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    if args.tf32 and not has_sm80:
        log_on_main("TF32 requires NVIDIA compute capability >= 8. Setting tf32 to False.")
        args.tf32 = False

    training_kwargs = dict(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="linear",
        warmup_steps=args.warmup_steps,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        logging_strategy="steps",
        logging_steps=10,
        disable_tqdm=False,
        report_to="none",
        max_steps=num_steps,
        max_grad_norm=1.0,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        tf32=args.tf32,
        bf16=args.bf16,
        save_only_model=False,
        prediction_loss_only=True,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy != "no" else None,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        label_names=["future_values", "future_values_mask"],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = TimesFMTrainer(
        model=model,
        args=TrainingArguments(**training_kwargs),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        point_loss=args.point_loss,
        huber_beta=args.huber_beta,
        quantile_loss_weight=args.quantile_loss_weight,
    )

    trainer.train(resume_from_checkpoint=args.resume_checkpoint)

    if is_main_process():
        final_ckpt = output_dir / "checkpoint-final"
        trainer.save_model(str(final_ckpt))
        trainer.save_state()
        if trainer.optimizer is not None:
            torch.save(trainer.optimizer.state_dict(), final_ckpt / "optimizer.pt")
        if trainer.lr_scheduler is not None:
            torch.save(trainer.lr_scheduler.state_dict(), final_ckpt / "scheduler.pt")
        training_info_config = deepcopy(raw_config)
        training_info_config.update(
            {
                "training_data_paths": data_paths_list,
                "context_length": args.context_length,
                "prediction_length": args.prediction_length,
                "min_past": args.min_past,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "learning_rate": args.learning_rate,
                "warmup_steps": args.warmup_steps,
                "num_epochs": args.num_epochs,
                "num_steps": num_steps,
                "train_target_series": train_target_series,
                "val_target_series": val_target_series,
                "point_loss": args.point_loss,
                "huber_beta": args.huber_beta,
                "quantile_loss_weight": args.quantile_loss_weight,
                "covariates_used_for_training": False,
            }
        )
        save_training_info(final_ckpt, training_config=training_info_config)


if __name__ == "__main__":
    main()


# trian with dataset A
# torchrun --nproc_per_node=8 train.py --training-data-paths ../../autodl-tmp/A --num-epochs 1 --point-loss huber --huber-beta 1.0

# train with datasets A+B
# torchrun --nproc_per_node=8 train.py --training-data-paths ../dataset/A+../dataset/B --num-epochs 1 --point-loss huber --huber-beta 1.0

