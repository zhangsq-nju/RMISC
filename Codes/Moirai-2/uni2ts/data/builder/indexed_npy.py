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

from dataclasses import dataclass
from inspect import signature
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from uni2ts.common.typing import Data, FlattenedData
from uni2ts.data.builder._base import DatasetBuilder
from uni2ts.data.dataset import SampleTimeSeriesType, TimeSeriesDataset
from uni2ts.transform import Transformation


def _to_uni2ts_series(values: np.ndarray) -> np.ndarray:
    """
    Convert time-major arrays to uni2ts layout.

    uni2ts uses (time,) for univariate series and (variate, time) for
    multivariate series.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        return values
    if values.shape[1] == 1:
        return values[:, 0]
    return values.T



def _flatten_data(data: dict[str, Data]) -> dict[str, FlattenedData]:
    return TimeSeriesDataset._flatten_data(data)


class IndexedNpyDataset(IterableDataset):
    """
    Streams sampled windows from npy files using parsed DataIndex rows.

    DataIndex is parsed in train.py and passed in as index_rows. Sharding across
    DDP ranks and DataLoader workers is handled in __iter__.
    """

    def __init__(
        self,
        data_path: Path,
        index_file: Path,
        transform: Transformation,
        random_crop: bool = False,
        shuffle: bool = False,
        array_cache_size: int = 16,
        prediction_length: Optional[int] = None,
        patch_size: Optional[int] = None,
        index_rows: Optional[list[tuple[str, str, int, int, int, list[int]]]] = None,
    ):
        self.data_path = Path(data_path)
        self.index_file = Path(index_file)
        self.transform = transform
        self.random_crop = random_crop
        self.shuffle = shuffle
        self.array_cache_size = array_cache_size
        self.min_past = 64  ### 加入了 self.min_past，固定最短 crop 长度
        self.prediction_length = prediction_length
        self.patch_size = patch_size
        self._array_cache: dict[tuple[str, str, int], np.ndarray] = {}
        self._epoch = 0

        if not self.data_path.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        if not self.index_file.is_file():
            raise FileNotFoundError(f"DataIndex file not found: {self.index_file}")

        if index_rows is None:
            raise ValueError(
                "IndexedNpyDataset requires index_rows from train.py. "
                "Do not read DataIndex CSV inside indexed_npy.py."
            )
        self.rows = index_rows
        if not self.rows:
            raise ValueError(f"DataIndex rows are empty for: {self.index_file}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rng = self._get_rng()
        if self.shuffle:
            ### 修改了shuffle随机种子，目的是train每个epoch使用不同但各rank一致的整epoch shuffle顺序。
            rng = np.random.default_rng(self._epoch)
            self._epoch += 1
        rows = self._iter_sharded_rows(rng)

        for idx, row in rows:
            yield self._make_sample(row, idx, rng)

    def _iter_sharded_rows(self, rng: np.random.Generator) -> Iterator[tuple[int, Any]]:
        shard_id, num_shards = self._get_shard_info()
        ### 先对全局idx做epoch级shuffle，再按rank/worker切片，避免固定idx % num_shards分布偏差。
        index_dtype = np.int32 if len(self.rows) <= np.iinfo(np.int32).max else np.int64
        row_indices = np.arange(len(self.rows), dtype=index_dtype)
        if self.shuffle:
            rng.shuffle(row_indices)
        row_indices = row_indices[shard_id::num_shards]
        for idx in row_indices:  ### 从内存中的 CSV list 按 rank/worker 切分
            yield int(idx), self.rows[int(idx)]

    @staticmethod
    def _get_rng() -> np.random.Generator:
        worker_info = get_worker_info()
        seed = worker_info.seed if worker_info is not None else torch.initial_seed()
        return np.random.default_rng(seed % 2**32)

    def _get_shard_info(self) -> tuple[int, int]:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        return rank * num_workers + worker_id, world_size * num_workers

    def _make_sample(
        self,
        row: Any,
        idx: int,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        if not isinstance(row, tuple):
            raise TypeError(
                "IndexedNpyDataset only accepts parsed tuple rows from train.py."
            )
        (
            data_root,
            dataset_name,
            numpy_id,
            time_start,
            time_stop,
            target,
        ) = row
        data_path = Path(data_root) / "Data"
        array = self._load_array(str(dataset_name), int(numpy_id), data_path)

        if time_start < 0 or time_stop > len(array) or time_start >= time_stop:
            raise ValueError(
                f"Invalid time range [{time_start}, {time_stop}) for "
                f"{dataset_name}/{numpy_id}.npy with length {len(array)}"
            )

        if not target:
            raise ValueError(
                f"Row {idx} in {self.index_file} has no target columns."
            )
        if len(target) != 1:
            raise ValueError(
                f"Row {idx} in {self.index_file} must contain exactly one target "
                f"column after train.py expansion, got {target}."
            )
        target_col = int(target[0])

        window_length = time_stop - time_start  ### 区分短序列 pad 和长序列 crop
        should_pad_short = window_length < 128  ### 长度小于 128 时不做 crop
        if not should_pad_short and self.random_crop:
            time_start = self._random_crop(time_start, time_stop, rng)
        window = array[time_start:time_stop, target_col]
        if should_pad_short:
            window = self._pad_short_window(window)  ### 长度小于 128 时在最后补 ceil(len/2) 个 NaN
        window = self._pad_to_patch_size(window)  ### 在进入 PatchCrop/Patchify 前保证长度是 patch_size 的整数倍
        data: dict[str, Data] = {
            "target": _to_uni2ts_series(window),
        }

        return self.transform(_flatten_data(data))

    def _pad_short_window(self, window: np.ndarray) -> np.ndarray:
        pad_length = self.prediction_length - int(np.floor(len(window) / 2)) 
        pad = np.full((pad_length,) + window.shape[1:], np.nan, dtype=np.float32)
        return np.concatenate([window.astype(np.float32, copy=False), pad], axis=0)

    def _pad_to_patch_size(self, window: np.ndarray) -> np.ndarray:
        if self.patch_size is None:
            return window
        pad_length = -len(window) % self.patch_size
        if pad_length == 0:
            return window
        pad = np.full((pad_length,) + window.shape[1:], np.nan, dtype=np.float32)
        return np.concatenate([pad, window.astype(np.float32, copy=False)], axis=0)

    def _random_crop(
        self,
        time_start: int,
        time_stop: int,
        rng: np.random.Generator,
    ) -> int:
        window_length = time_stop - time_start
        if window_length < 128:
            return time_start

        crop_start = time_start + int(
            rng.integers(window_length - self.min_past - self.prediction_length + 1)
        )
        return crop_start

    def _load_array(
        self, dataset_name: str, numpy_id: int, data_path: Optional[Path] = None
    ) -> np.ndarray:
        base_data_path = Path(data_path) if data_path is not None else self.data_path
        cache_key = (str(base_data_path), dataset_name, numpy_id)
        if cache_key in self._array_cache:
            return self._array_cache[cache_key]

        file_path = base_data_path / dataset_name / f"{numpy_id}.npy"
        if not file_path.is_file():
            raise FileNotFoundError(f"npy data file not found: {file_path}")

        array = np.load(file_path, mmap_mode="r")
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim > 2:
            raise ValueError(f"Expected 1-D or 2-D npy array at {file_path}, got {array.shape}")

        if self.array_cache_size > 0:
            if len(self._array_cache) >= self.array_cache_size:
                self._array_cache.pop(next(iter(self._array_cache)))
            self._array_cache[cache_key] = array

        return array


@dataclass
class IndexedNpyDatasetBuilder(DatasetBuilder):
    """
    DatasetBuilder for datasets stored as npy arrays plus sampled window indices.

    Expected layout:
        root_path / Data / <dataset> / <numpy>.npy
        root_path / DataIndex / <split> / <dataset>.csv

    Each DataIndex row is treated as one training/evaluation sample. Multi-card
    parallelism is handled by the existing DistributedSampler in cli/train.py.
    """

    dataset: str
    split: str = "train"
    root_path: Optional[Path] = None
    data_path: Optional[Path] = None
    index_path: Optional[Path] = None
    weight: float = 1.0
    sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE
    shuffle: bool = True  ### 修改了Builder配置字段，目的是用shuffle替代shuffle_buffer_size。
    array_cache_size: int = 16
    index_rows: Optional[list[tuple[str, str, int, int, int, list[int]]]] = None
    transform_kwargs: Optional[dict[str, Any]] = None
    distance: Optional[int] = None
    prediction_length: Optional[int] = None
    context_length: Optional[int] = None
    patch_size: Optional[int] = None
    offset: Optional[int] = None
    storage_path: Optional[Path] = None
    output_prefix: str = "indexed_npy"

    def __post_init__(self):
        self.storage_path = Path(self.storage_path) if self.storage_path else None
        self.root_path = self._resolve_root_path()
        self.data_path = (
            Path(self.data_path) if self.data_path else self.root_path / "Data"
        )
        self.index_path = (
            Path(self.index_path) if self.index_path else self.root_path / "DataIndex"
        )

        if self.sample_time_series != SampleTimeSeriesType.NONE:
            raise ValueError(
                "IndexedNpyDataset rows are already sampled by DataIndex; use "
                "DataLoader shuffle/DistributedSampler instead of sample_time_series."
            )

    @property
    def index_file(self) -> Path:
        return self.index_path / self.split / f"{self.dataset}.csv"

    @property
    def dataset_path(self) -> Path:
        return self.storage_path / self.output_prefix / self.split / self.dataset

    def build_dataset(self):
        """
        Validate the on-disk layout.

        Unlike SimpleDatasetBuilder, this builder intentionally does not materialize
        windows into a Hugging Face dataset. The DataIndex CSV is already the sample
        list, and lazy npy loading is friendlier to multi-GPU/multi-worker training.
        """
        if not self.data_path.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")
        if not self.index_file.is_file():
            raise FileNotFoundError(f"DataIndex file not found: {self.index_file}")


    def load_dataset(
        self, transform_map: dict[str | type, Callable[..., Transformation]]
    ) -> Dataset:
        transform_func = self._get_transform_func(transform_map)
        transform_kwargs = self._get_transform_kwargs(transform_func)

        return IndexedNpyDataset(
            data_path=self.data_path,
            index_file=self.index_file,
            transform=transform_func(**transform_kwargs),
            random_crop=True if self.split == "train" else False, ### 训练时随机裁剪，评估时不裁剪
            ### 修改了Dataset shuffle传参，目的是由train.yaml/val.yaml的shuffle决定是否整epoch shuffle。
            shuffle=self.shuffle,
            array_cache_size=self.array_cache_size,
            prediction_length=self.prediction_length,
            patch_size=self.patch_size,
            index_rows=self.index_rows,
        )

    def _resolve_root_path(self) -> Path:
        if self.root_path is not None:
            return Path(self.root_path)
        else:
            raise ValueError("root_path must be provided for IndexedNpyDatasetBuilder.")


    def _get_transform_func(
        self, transform_map: dict[str | type, Callable[..., Transformation]]
    ) -> Callable[..., Transformation]:
        if self.dataset in transform_map:
            return transform_map[self.dataset]
        if "default" in transform_map:
            return transform_map["default"]

        try:
            return transform_map[self.dataset]
        except KeyError as error:
            raise KeyError(
                f"No transform found for dataset {self.dataset}. Provide either "
                f"'{self.dataset}' or 'default' in transform_map."
            ) from error

    def _get_transform_kwargs(
        self, transform_func: Callable[..., Transformation]
    ) -> dict[str, Any]:
        explicit_kwargs = self.transform_kwargs or {}
        params = signature(transform_func).parameters
        inferred_kwargs: dict[str, Any] = {
            "distance": self.distance,
            "prediction_length": self.prediction_length,
            "context_length": self.context_length,
            "patch_size": self.patch_size,
            "offset": self.offset if self.offset is not None else self.context_length,
        }
        kwargs = {
            key: value
            for key, value in inferred_kwargs.items()
            if key in params and value is not None
        }
        kwargs |= explicit_kwargs

        missing = [
            key
            for key, param in params.items()
            if param.default is param.empty and key not in kwargs
        ]
        if missing:
            raise ValueError(
                f"Missing transform arguments for {self.dataset}: {missing}. "
                "Set them directly or via transform_kwargs on IndexedNpyDatasetBuilder."
            )

        return kwargs
