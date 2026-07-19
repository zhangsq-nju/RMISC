import ast
import glob
import math
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset


@dataclass(frozen=True)
class IndexRow:
    data_root: str
    dataset: str
    numpy_idx: int
    time_start: int
    time_stop: int
    target: tuple[int, ...]
    cov: tuple[int, ...]

    @property
    def length(self) -> int:
        return self.time_stop - self.time_start

    @property
    def target_width(self) -> int:
        return len(self.target)


def resolve_data_paths(training_data_paths: str) -> list[str]:
    raw_paths = re.split(r"[,+]", training_data_paths)
    data_paths: list[str] = []
    for raw_path in raw_paths:
        raw_path = raw_path.strip()
        if not raw_path:
            continue

        candidates = [
            Path(raw_path),
            Path("dataset") / raw_path,
            Path("data") / raw_path,
        ]
        data_path = next((candidate for candidate in candidates if candidate.exists()), Path(raw_path))

        if not (data_path / "Data").is_dir():
            raise FileNotFoundError(f"找不到数据目录: {data_path / 'Data'}")
        if not (data_path / "DataIndex").is_dir():
            raise FileNotFoundError(f"找不到索引目录: {data_path / 'DataIndex'}")
        data_paths.append(str(data_path))

    if len(data_paths) == 0:
        raise ValueError("training_data_paths 不能为空")
    return data_paths


def _parse_list(value) -> tuple[int, ...]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    return tuple(int(x) for x in value)


def _read_index_file(file_path: str, data_root: str) -> list[IndexRow]:
    df = pd.read_csv(
        file_path,
        usecols=["dataset", "numpy", "time_start", "time_stop", "target", "cov"],
    )

    rows: list[IndexRow] = []
    for row in df.itertuples(index=False):
        rows.append(
            IndexRow(
                data_root=str(data_root),
                dataset=str(row.dataset),
                numpy_idx=int(row.numpy),
                time_start=int(row.time_start),
                time_stop=int(row.time_stop),
                target=_parse_list(row.target),
                cov=_parse_list(row.cov),
            )
        )
    return rows


def input_convert(datapaths: str | Sequence[str], flag: str) -> list[IndexRow]:
    datapath_list = list(datapaths) if isinstance(datapaths, (list, tuple)) else [datapaths]
    index: list[IndexRow] = []

    for one_datapath in datapath_list:
        files = sorted(glob.glob(f"{one_datapath}/DataIndex/{flag}/*.csv"))
        if len(files) == 0:
            print(f"Warning: 找不到索引文件: {one_datapath}/DataIndex/{flag}/*.csv")
        for file in files:
            print(file)
            index.extend(_read_index_file(file, str(one_datapath)))
    return index


def filter_rows(rows: Sequence[IndexRow], min_length: int) -> list[IndexRow]:
    return [row for row in rows if row.length >= min_length and row.target_width > 0]


def count_target_series(rows: Sequence[IndexRow]) -> int:
    return sum(row.target_width for row in rows)


class TimesFMIndexedDataset(IterableDataset):
    """Iterable dataset that trains TimesFM only on univariate target columns.

    The `cov` field from DataIndex is intentionally ignored here. Covariates are
    only needed later by external XReg evaluation code, never by training loss.
    """

    def __init__(
        self,
        rows: Sequence[IndexRow],
        context_length: int,
        prediction_length: int,
        batch_size: int,
        min_past: int,
        mode: str,
        repeat: bool | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"train", "val", "test"}:
            raise ValueError(f"Unexpected mode: {mode}")

        self.rows = list(filter_rows(rows, min_length=min_past + prediction_length))
        if len(self.rows) == 0:
            raise ValueError("过滤后数据为空，请检查序列长度、target 列和 min_past/prediction_length。")

        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.batch_size = int(batch_size)
        self.min_past = int(min_past)
        self.mode = mode
        self.repeat = (mode == "train") if repeat is None else bool(repeat)

        self._array_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._array_cache_limit = 16

    @property
    def target_series_count(self) -> int:
        return count_target_series(self.rows)

    def _get_cached_array(self, npy_path: str) -> np.ndarray:
        cached = self._array_cache.get(npy_path)
        if cached is not None:
            self._array_cache.move_to_end(npy_path)
            return cached

        array = np.load(npy_path, mmap_mode="r")
        if array.ndim == 1:
            array = array[:, None]
        self._array_cache[npy_path] = array
        if len(self._array_cache) > self._array_cache_limit:
            self._array_cache.popitem(last=False)
        return array

    def _read_series(self, row: IndexRow, target_col: int) -> np.ndarray:
        npy_path = f"{row.data_root}/Data/{row.dataset}/{row.numpy_idx}.npy"
        array = self._get_cached_array(npy_path)
        series = array[row.time_start : row.time_stop, target_col].astype(np.float32, copy=False)
        return series

    def _normalize_window(
            self,
            context: np.ndarray,
            future: np.ndarray,
            epsilon: float = 1e-5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        context = np.asarray(context, dtype=np.float32)
        future = np.asarray(future, dtype=np.float32)

        context_observed = ~np.isnan(context)
        if context_observed.any():
            loc = np.nanmean(context)
            loc = np.nan_to_num(loc, nan=0.0)
            scale = np.sqrt(np.nanmean(np.square(context - loc)))
            scale = np.nan_to_num(scale, nan=1.0)
        else:
            loc = np.float32(0.0)
            scale = np.float32(1.0)

        if scale == 0.0:
            scale = np.float32(epsilon)

        context = (context - loc) / scale
        future = (future - loc) / scale

        # Match Chronos InstanceNorm with use_arcsinh=True.
        context = np.arcsinh(context)
        future = np.arcsinh(future)

        context = np.clip(context, -5.0, 5.0)
        future = np.clip(future, -5.0, 5.0)

        future_mask = (~np.isnan(future)).astype(np.float32, copy=False)
        context = np.nan_to_num(context, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32, copy=False)
        future = np.nan_to_num(future, nan=0.0, posinf=5.0, neginf=-5.0).astype(np.float32, copy=False)

        return context, future, future_mask

    def _construct_sample(self, row_idx: int, target_pos: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[row_idx]
        target_col = row.target[target_pos]
        series = self._read_series(row, target_col)
        full_length = len(series)

        if self.mode == "train":
            max_begin = full_length - self.prediction_length - self.min_past
            begin_idx = np.random.randint(0, max_begin + 1)
            slice_idx = full_length - self.prediction_length
        else:
            begin_idx = 0
            slice_idx = full_length - self.prediction_length

        if slice_idx - begin_idx >= self.context_length:
            context = series[slice_idx - self.context_length : slice_idx]
        else:
            context = series[begin_idx:slice_idx]

        future = series[slice_idx : slice_idx + self.prediction_length]
        context, future, future_mask = self._normalize_window(context, future)
        return (
            torch.from_numpy(context.copy()),
            torch.from_numpy(future.copy()),
            torch.from_numpy(future_mask.copy()),
        )

    def _build_batch(self, pairs: Sequence[tuple[int, int]]) -> dict:
        past_values: list[torch.Tensor] = []
        future_values: list[torch.Tensor] = []
        future_values_mask: list[torch.Tensor] = []
        for row_idx, target_pos in pairs:
            context, future, future_mask = self._construct_sample(row_idx, target_pos)
            if self.mode == "train" and future_mask.sum().item() == 0:
                continue
            past_values.append(context)
            future_values.append(future)
            future_values_mask.append(future_mask)

        if len(past_values) == 0:
            context = torch.zeros(self.min_past, dtype=torch.float32)
            future = torch.zeros(self.prediction_length, dtype=torch.float32)
            future_mask = torch.zeros(self.prediction_length, dtype=torch.float32)
            past_values.append(context)
            future_values.append(future)
            future_values_mask.append(future_mask)

        return {
            "past_values": past_values,
            "future_values": torch.stack(future_values, dim=0),
            "future_values_mask": torch.stack(future_values_mask, dim=0),
            "forecast_context_len": self.context_length,
            "force_flip_invariance": False,
            "truncate_negative": False,
        }

    def _rank_info(self) -> tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _generate_train_batches(self) -> Iterator[dict]:
        rank, world_size = self._rank_info()
        base_seed = int(torch.initial_seed() % 2**32)
        epoch = 0

        while True:
            epoch_seed = base_seed + epoch
            random.Random(epoch_seed).shuffle(self.rows)
            rng = np.random.RandomState(seed=epoch_seed)
            pairs: list[tuple[int, int]] = []
            flat_idx = 0

            for row_idx, row in enumerate(self.rows):
                target_order = rng.permutation(row.target_width)
                for target_pos in target_order:
                    if flat_idx % world_size == rank:
                        pairs.append((row_idx, int(target_pos)))
                        if len(pairs) >= self.batch_size:
                            yield self._build_batch(pairs)
                            pairs = []
                    flat_idx += 1

            if pairs:
                yield self._build_batch(pairs)
            if not self.repeat:
                break
            epoch += 1

    def _generate_sequential_batches(self) -> Iterator[dict]:
        rank, world_size = self._rank_info()
        max_batches = None
        if self.mode == "val" and world_size > 1:
            total = self.target_series_count
            per_rank_batches = []
            for other_rank in range(world_size):
                count = sum(1 for flat_idx in range(total) if flat_idx % world_size == other_rank)
                per_rank_batches.append(math.ceil(count / self.batch_size))
            max_batches = min(per_rank_batches)

        pairs: list[tuple[int, int]] = []
        flat_idx = 0
        yielded_batches = 0

        for row_idx, row in enumerate(self.rows):
            for target_pos in range(row.target_width):
                if flat_idx % world_size == rank:
                    pairs.append((row_idx, target_pos))
                    if len(pairs) >= self.batch_size:
                        if max_batches is not None and yielded_batches >= max_batches:
                            return
                        yield self._build_batch(pairs)
                        yielded_batches += 1
                        pairs = []
                flat_idx += 1

        if pairs and (max_batches is None or yielded_batches < max_batches):
            yield self._build_batch(pairs)

    def __iter__(self) -> Iterator[dict]:
        if self.mode == "train":
            yield from self._generate_train_batches()
        else:
            while True:
                yielded = False
                for batch in self._generate_sequential_batches():
                    yielded = True
                    yield batch
                if not self.repeat or not yielded:
                    break


class MixedMultiRootDataset(IterableDataset):
    """Mix per-root iterable datasets so A+B does not consume all of A first."""

    def __init__(
        self,
        datasets: Sequence[TimesFMIndexedDataset],
        batches_per_dataset: Sequence[int],
        repeat: bool,
        shuffle: bool,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.datasets = list(datasets)
        self.batches_per_dataset = [int(x) for x in batches_per_dataset]
        self.repeat = bool(repeat)
        self.shuffle = bool(shuffle)
        self.batch_size = int(batch_size)

        if len(self.datasets) != len(self.batches_per_dataset):
            raise ValueError("datasets 和 batches_per_dataset 长度不一致。")
        if len(self.datasets) == 0:
            raise ValueError("MixedMultiRootDataset 至少需要一个子数据集。")

    def __iter__(self) -> Iterator[dict]:
        rng = np.random.default_rng()

        while True:
            iterators = [iter(dataset) for dataset in self.datasets]
            order: list[int] = []
            for dataset_idx, max_batches in enumerate(self.batches_per_dataset):
                order.extend([dataset_idx] * max(0, max_batches))
            if self.shuffle:
                rng.shuffle(order)

            yielded_any = False
            for dataset_idx in order:
                try:
                    batch = next(iterators[dataset_idx])
                except StopIteration:
                    continue
                yielded_any = True
                yield batch

            if not self.repeat or not yielded_any:
                break


def make_dataset(
    data_paths: str | Sequence[str],
    rows: Sequence[IndexRow],
    flag: str,
    context_length: int,
    prediction_length: int,
    batch_size: int,
    min_past: int,
    mode: str,
) -> IterableDataset:
    if not isinstance(data_paths, (list, tuple)):
        return TimesFMIndexedDataset(
            rows=rows,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            min_past=min_past,
            mode=mode,
        )

    datasets: list[TimesFMIndexedDataset] = []
    batches_per_dataset: list[int] = []
    for one_data_path in data_paths:
        one_rows = [row for row in rows if row.data_root == str(one_data_path)]
        if len(one_rows) == 0:
            continue
        dataset = TimesFMIndexedDataset(
            rows=one_rows,
            context_length=context_length,
            prediction_length=prediction_length,
            batch_size=batch_size,
            min_past=min_past,
            mode=mode,
        )
        datasets.append(dataset)
        batches_per_dataset.append(max(1, math.ceil(dataset.target_series_count / batch_size)))

    if len(datasets) == 0:
        raise ValueError(f"{flag}_dataset 为空，请检查 DataIndex/{flag}/*.csv。")

    return MixedMultiRootDataset(
        datasets=datasets,
        batches_per_dataset=batches_per_dataset,
        repeat=(mode == "train"),
        shuffle=(mode == "train"),
        batch_size=batch_size,
    )
