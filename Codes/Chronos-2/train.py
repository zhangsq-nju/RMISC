import logging
import os
import re
import sys
import json
import math
import ast
import itertools
import random
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import List, Iterator, Optional, Dict

import typer
from typer_config import use_yaml_config
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
import transformers
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    T5Config,
    Trainer,
    TrainingArguments,
)
import accelerate
import gluonts
from chronos.chronos2 import Chronos2Pipeline,Chronos2Model,Chronos2CoreConfig
from chronos.chronos2.train_line import Train
from chronos import BaseChronosPipeline, Chronos2Pipeline

from collections import OrderedDict
from transformers.modeling_utils import load_state_dict

app = typer.Typer(pretty_exceptions_enable=False)
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "10000"


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


def setup_output_capture(output_dir: Path):
    if not is_main_process():
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    log_fp = open(output_dir / "train.log", "a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_fp)
    sys.stderr = _TeeStream(sys.stderr, log_fp)
    return log_fp


def load_weights(dict_path):
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(dict_path, device_map="cpu")
    model = pipeline.model
    state_dict = load_state_dict(f"{dict_path}/model.safetensors")
    if any(k.startswith("module.") for k in state_dict.keys()):
        new_state_dict = OrderedDict(
            (k.replace("module.", "", 1), v) for k, v in state_dict.items()
        )
        state_dict = new_state_dict

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")
    return model

def is_main_process() -> bool:
    """
    Check if we're on the main process.
    """
    return int(os.environ.get("RANK", 0)) == 0


def get_shared_output_dir(base_dir: Path) -> Path:
    """
    Let rank 0 choose the run directory once, then share it with all ranks.
    """
    base_dir.mkdir(parents=True, exist_ok=True)

    if not (dist.is_available() and dist.is_initialized()):
        return get_next_path("run", base_dir=base_dir, file_type="")

    output_dir_holder = [
        str(get_next_path("run", base_dir=base_dir, file_type=""))
        if is_main_process()
        else None
    ]
    dist.broadcast_object_list(output_dir_holder, src=0)
    return Path(output_dir_holder[0])


def log_on_main(msg: str, logger: logging.Logger, log_level: int = logging.INFO):
    """
    Log the given message using the given logger, if we're on the main process.
    """
    if is_main_process():
        logger.log(log_level, msg)


def get_training_job_info() -> Dict:
    """
    Returns info about this training job.
    """
    job_info = {}

    # CUDA info
    job_info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        job_info["device_count"] = torch.cuda.device_count()

        job_info["device_names"] = {
            idx: torch.cuda.get_device_name(idx)
            for idx in range(torch.cuda.device_count())
        }
        job_info["mem_info"] = {
            idx: torch.cuda.mem_get_info(device=idx)
            for idx in range(torch.cuda.device_count())
        }

    # DDP info
    job_info["torchelastic_launched"] = dist.is_torchelastic_launched()

    if dist.is_torchelastic_launched():
        job_info["world_size"] = dist.get_world_size()

    # Versions
    job_info["python_version"] = sys.version.replace("\n", " ")
    job_info["torch_version"] = torch.__version__
    job_info["numpy_version"] = np.__version__
    job_info["gluonts_version"] = gluonts.__version__
    job_info["accelerate_version"] = accelerate.__version__

    return job_info


def save_training_info(ckpt_path: Path, training_config: Dict):
    """
    Save info about this training job in a json file for documentation.
    """
    assert ckpt_path.is_dir()
    with open(ckpt_path / "training_info.json", "w") as fp:
        json.dump(
            {"training_config": training_config, "job_info": get_training_job_info()},
            fp,
            indent=4,
        )


def get_next_path(
    base_fname: str,
    base_dir: Path,
    file_type: str = "yaml",
    separator: str = "-",
):
    """
    Gets the next available path in a directory. For example, if `base_fname="results"`
    and `base_dir` has files ["results-0.yaml", "results-1.yaml"], this function returns
    "results-2.yaml".
    """
    if file_type == "":
        # Directory
        items = filter(
            lambda x: x.is_dir() and re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob("*"),
        )
    else:
        # File
        items = filter(
            lambda x: re.match(f"^{base_fname}{separator}\\d+$", x.stem),
            base_dir.glob(f"*.{file_type}"),
        )
    run_nums = list(
        map(lambda x: int(x.stem.replace(base_fname + separator, "")), items)
    ) + [-1]

    next_num = max(run_nums) + 1
    fname = f"{base_fname}{separator}{next_num}" + (
        f".{file_type}" if file_type != "" else ""
    )

    return base_dir / fname


import glob
import pandas as pd
def resolve_data_paths(training_data_paths):
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
        ]
        data_path = None
        for candidate in candidates:
            if candidate.exists():
                data_path = candidate
                break
        if data_path is None:
            data_path = Path(raw_path)

        if not (data_path / "Data").is_dir():
            raise FileNotFoundError(f"找不到数据目录: {data_path / 'Data'}")
        if not (data_path / "DataIndex").is_dir():
            raise FileNotFoundError(f"找不到索引目录: {data_path / 'DataIndex'}")

        data_paths.append(str(data_path))

    if len(data_paths) == 0:
        raise ValueError("training_data_paths 不能为空")
    return data_paths

def _read_index_file(file_path: str, data_root: str) -> list[tuple]:
    """
    Read one DataIndex csv and eagerly parse fields that would otherwise be
    repeatedly parsed inside the training loop.
    """
    df = pd.read_csv(
        file_path,
        usecols=["dataset", "numpy", "time_start", "time_stop", "target", "cov"],
    )

    rows = []
    for row in df.itertuples(index=False):
        target = ast.literal_eval(row.target) if isinstance(row.target, str) else row.target
        cov = ast.literal_eval(row.cov) if isinstance(row.cov, str) else row.cov
        rows.append(
            (
                str(data_root),
                str(row.dataset),
                int(row.numpy),
                int(row.time_start),
                int(row.time_stop),
                list(target),
                list(cov),
            )
        )
    return rows


def InputConvert(datapath, flag):
    datapaths = list(datapath) if isinstance(datapath, (list, tuple)) else [datapath]
    index = []

    for one_datapath in datapaths:
        files = sorted(glob.glob(f"{one_datapath}/DataIndex/{flag}/*.csv"))
        if len(files) == 0:
            print(f"Warning: 找不到索引文件: {one_datapath}/DataIndex/{flag}/*.csv")
        for file in files:
            print(file)
            index.extend(_read_index_file(file, one_datapath))
    return index


def average_input_width(inputs) -> float:
    if len(inputs) == 0:
        return 0.0

    total_width = 0
    for item in inputs:
        if isinstance(item, (list, tuple)) and len(item) >= 7:
            target = item[5]
            cov = item[6]
            total_width += len(target) + len(cov)
        else:
            target = item["target"]
            target_width = 1 if np.asarray(target).ndim == 1 else np.asarray(target).shape[0]
            past_covariates = item.get("past_covariates", {})
            total_width += target_width + len(past_covariates)

    return total_width / len(inputs)


@app.command()
@use_yaml_config(param_name="config")
def main(
    training_data_paths: str="data/large",
    context_length: int = 1984,
    prediction_length: int = 64,
    min_past: int = 64,
    per_device_train_batch_size: int = 128,
    gradient_accumulation_steps: int = 12, 
    num_steps: int = 200000,
    num_epochs: int = 0, 
    save_steps: int = 2000, 
    eval_steps: int = 2000, 
    output_dir: str = "../output/",
    resume_checkpoint: str = None,
    tf32: bool = True,
    local_rank: int = -1,
):
    if torch.cuda.is_available():
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        global_rank = int(os.environ.get("RANK", 0))
        num_gpus = torch.cuda.device_count()
        local_rank = int(os.environ.get("LOCAL_RANK", global_rank % num_gpus))
        torch.cuda.set_device(local_rank)
        import datetime
        dist.init_process_group(
            backend="nccl",
            init_method='env://',
            world_size=world_size,
            rank=global_rank,
            timeout=datetime.timedelta(seconds=30 * 60)
        )

    if tf32 and not (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ):
        # TF32 floating point format is available only on NVIDIA GPUs
        # with compute capability 8 and above. See link for details.
        # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-8-x
        log_on_main(
            "TF32 format is only available on devices with compute capability >= 8. "
            "Setting tf32 to False.",
            logger,
        )
        tf32 = False

    output_dir = Path(output_dir)
    data_paths_list = resolve_data_paths(training_data_paths)
    data_paths = data_paths_list[0] if len(data_paths_list) == 1 else data_paths_list

    training_data_paths = []
    for data_path in data_paths_list:
        training_data_paths.extend(glob.glob(f"{data_path}/Data/**/*.npy", recursive=True))
    assert isinstance(training_data_paths, list)

    output_dir = get_shared_output_dir(output_dir)
    log_fp = setup_output_capture(output_dir)

    log_on_main(f"Logging dir: {output_dir}", logger)
    log_on_main(
        f"Loading and filtering {len(training_data_paths)} datasets for training",
        logger,
    )
    log_on_main(
        f"Resolved training data roots: {data_paths_list}",
        logger,
    )

    # Load Configs and init model
    log_on_main("Initializing model", logger)
    with open("config.json", 'r', encoding='utf-8') as f:
        jsconfig = json.load(f)
    config = Chronos2CoreConfig(
        d_model=jsconfig["d_model"],
        d_kv=jsconfig["d_kv"],
        d_ff=jsconfig["d_ff"],
        num_layers=jsconfig["num_layers"],
        num_heads=jsconfig["num_heads"],
        dropout_rate=jsconfig["dropout_rate"],
        layer_norm_epsilon=jsconfig["layer_norm_epsilon"],
        initializer_factor=jsconfig["initializer_factor"],
        feed_forward_proj=jsconfig["feed_forward_proj"],
        vocab_size=jsconfig["vocab_size"],
        pad_token_id=jsconfig["pad_token_id"],
        rope_theta=jsconfig["rope_theta"],
        attn_implementation=None,
    )

    config.chronos_config = jsconfig["chronos_config"]
    config.chronos_pipeline_class = "Chronos2Pipeline"
    model = Chronos2Model(config=config)
    
    if torch.cuda.is_available():
        model = model.cuda(local_rank)
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False
        )

    # Train Model
    train_inputs=InputConvert(data_paths,"train")
    val_inputs = InputConvert(data_paths, "val")

    if num_epochs > 0:
        if len(train_inputs) == 0:
            raise ValueError("train_inputs 为空，无法根据 num_epochs 计算 num_steps。请检查 DataIndex/train/*.csv。")
        world_size_for_steps = int(os.environ.get("WORLD_SIZE", 1))
        avg_input_width = average_input_width(train_inputs)
        effective_samples_per_step = (
            per_device_train_batch_size
            * world_size_for_steps
            * gradient_accumulation_steps
            / avg_input_width
        )
        steps_per_epoch = math.ceil(len(train_inputs) / effective_samples_per_step)
        num_steps = steps_per_epoch * num_epochs
        eval_steps = max(1, math.ceil(steps_per_epoch / 4))
        save_steps = eval_steps
        log_on_main(
            f"Using epoch-based stopping: num_epochs={num_epochs}, "
            f"len(train_inputs)={len(train_inputs)}, "
            f"avg_input_width={avg_input_width:.4f}, "
            f"world_size={world_size_for_steps}, "
            f"per_device_train_batch_size={per_device_train_batch_size}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"effective_samples_per_step={effective_samples_per_step:.4f}, "
            f"steps_per_epoch={steps_per_epoch}, "
            f"eval_steps={eval_steps}, "
            f"save_steps={save_steps}, "
            f"num_steps={num_steps}",
            logger,
        )
    else:
        log_on_main(
            f"Using step-based stopping: num_steps={num_steps}",
            logger,
        )

    train=Train(model)
    resume_checkpoint = resume_checkpoint
    if resume_checkpoint:
        output_dir = "/".join(resume_checkpoint.split("/")[:-1])
    train.fit(datapath=data_paths, data_paths=data_paths,inputs = train_inputs, prediction_length = prediction_length, validation_inputs = val_inputs, context_length = context_length, num_steps = num_steps, batch_size = per_device_train_batch_size, gradient_accumulation_steps = gradient_accumulation_steps, save_steps = save_steps, eval_steps = eval_steps, output_dir=output_dir, min_past = min_past, resume_checkpoint = resume_checkpoint)

    # Save Final Model
    if is_main_process():
        save_model = model.module if isinstance(model, DDP) else model
        save_model.save_pretrained(output_dir / "checkpoint-final")
        training_info_config = deepcopy(jsconfig)
        training_info_config["training_data_paths"] = data_paths_list
        training_info_config["num_epochs"] = num_epochs
        training_info_config["num_steps"] = num_steps
        save_training_info(
            output_dir / "checkpoint-final", training_config=training_info_config
        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)
    app()


# train with dataset A
# torchrun --nproc_per_node=8 train.py --training-data-paths ../dataset/A --num-epochs 1

# train with datasets A+B
# torchrun --nproc_per_node=8 train.py --training-data-paths ../dataset/A+../dataset/B --num-epochs 1
