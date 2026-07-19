import ast
import os
import random
import threading
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from utils.chronos_norm import denormalize as chronos_denormalize
from utils.chronos_norm import normalize_with_stats as chronos_normalize_with_stats
from utils.chronos_norm import observed_mask as chronos_observed_mask


INVALID_ABS_LIMIT = float(os.environ.get("GTT_INVALID_ABS_LIMIT", "1e19"))


class DataHandler:
    def __init__(self, block_size, pred_len):
        self.block_size = block_size
        self.pred_len = pred_len
        self._mmap_cache = OrderedDict()
        self._mmap_cache_max = 64
        self._mmap_cache_lock = threading.Lock()

    def normalization(self, data, context_length=None, epsilon=1e-5):
        data, _, _ = self.normalize_with_stats(data, context_length=context_length, epsilon=epsilon)
        return data

    @staticmethod
    def _observed_mask(data):
        return chronos_observed_mask(data, invalid_abs_limit=INVALID_ABS_LIMIT)

    def normalize_with_stats(self, data, context_length=None, epsilon=1e-5):
        if context_length is None:
            data_arr = np.asarray(data, dtype=np.float32)
            context_length = max(data_arr.shape[0] - self.pred_len, 1)
        return chronos_normalize_with_stats(
            data,
            context_length=context_length,
            epsilon=epsilon,
            use_arcsinh=True,
            invalid_abs_limit=INVALID_ABS_LIMIT,
        )

    @staticmethod
    def denormalize(data, mean, std, epsilon=1e-5):
        return chronos_denormalize(data, mean, std, use_arcsinh=True)

    def _get_memmap(self, path: str):
        path = os.path.abspath(os.path.expanduser(path))
        with self._mmap_cache_lock:
            mm = self._mmap_cache.get(path)
            if mm is not None:
                self._mmap_cache.move_to_end(path)
                return mm

        mm = np.load(path, mmap_mode="r")
        with self._mmap_cache_lock:
            cached = self._mmap_cache.get(path)
            if cached is not None:
                self._mmap_cache.move_to_end(path)
                return cached
            self._mmap_cache[path] = mm
            self._mmap_cache.move_to_end(path)
            if len(self._mmap_cache) > self._mmap_cache_max:
                self._mmap_cache.popitem(last=False)
            return mm

    def index_to_array(self, ind, training: bool = False):
        root, dataset, parquet, start, stop, target, cov = ind
        start = int(start)
        stop = int(stop)
        target_idx = ast.literal_eval(target) if isinstance(target, str) else list(target)
        cov_idx = ast.literal_eval(cov) if isinstance(cov, str) else list(cov)
        target_idx = [int(i) for i in target_idx] if target_idx else []
        cov_idx = [int(i) for i in cov_idx] if cov_idx else []
        feat_idx = target_idx + cov_idx
        seen = set()
        feat_idx = [i for i in feat_idx if not (i in seen or seen.add(i))]

        path = f"{root}/Data/{dataset}/{parquet}.npy"
        large = self._get_memmap(path)
        seq_sel = large[start:stop]
        if seq_sel.ndim == 1:
            seq_sel = seq_sel.reshape(-1, 1)
        seq_sel = seq_sel[:, feat_idx] if len(feat_idx) > 0 else seq_sel[:, :0]

        full_length = seq_sel.shape[0]
        pred_len = self.pred_len
        context_length = self.block_size
        min_past = pred_len

        if full_length >= 128:
            slice_idx = full_length - pred_len
            if training:
                max_begin = full_length - pred_len - min_past
                begin_idx = np.random.randint(0, max_begin + 1) if max_begin > 0 else 0
            else:
                begin_idx = 0
        else:
            begin_idx = 0
            slice_idx = (full_length + 1) // 2

        if slice_idx - begin_idx >= context_length:
            context = seq_sel[slice_idx - context_length: slice_idx]
        else:
            context = seq_sel[begin_idx: slice_idx]

        future = seq_sel[slice_idx: full_length]
        if future.shape[0] > pred_len:
            future = future[:pred_len]
        future_len = future.shape[0]
        future_observed = self._observed_mask(future)
        future_raw = np.where(future_observed, future, np.nan)

        seq_sel = np.concatenate([context, future], axis=0)
        context_len = context.shape[0]
        seq_sel, mean, std = self.normalize_with_stats(seq_sel, context_length=context_len)
        context = seq_sel[:context_len]
        future = seq_sel[context_len:]

        f_max = 24
        k = min(len(feat_idx), f_max)
        x = np.zeros((self.block_size, f_max), dtype=np.float32)
        y = np.zeros((pred_len, f_max), dtype=np.float32)
        y_raw = np.zeros((pred_len, f_max), dtype=np.float32)
        if k > 0:
            context_pack = context[:, :k].astype(np.float32)
            future_pack = future[:, :k].astype(np.float32)
            future_raw_pack = future_raw[:, :k].astype(np.float32)
            context_take = min(context_pack.shape[0], self.block_size)
            future_take = min(future_pack.shape[0], pred_len)
            if context_take > 0:
                x[-context_take:, :k] = context_pack[-context_take:]
            if future_take > 0:
                y[:future_take, :k] = future_pack[:future_take]
                y_raw[:future_take, :k] = np.nan_to_num(future_raw_pack[:future_take], nan=0.0)

        target_mask = np.zeros((pred_len, f_max), dtype=np.float32)
        t = min(len(target_idx), f_max)
        if t > 0 and future_len > 0:
            valid_len = min(future_len, pred_len)
            target_mask[:valid_len, :t] = future_observed[:valid_len, :t].astype(np.float32)
        return x, y, target_mask

    def index_to_array_with_stats(self, ind, training: bool = False):
        x, y, target_mask = self.index_to_array(ind, training=training)

        root, dataset, parquet, start, stop, target, cov = ind
        start = int(start)
        stop = int(stop)
        target_idx = ast.literal_eval(target) if isinstance(target, str) else list(target)
        cov_idx = ast.literal_eval(cov) if isinstance(cov, str) else list(cov)
        target_idx = [int(i) for i in target_idx] if target_idx else []
        cov_idx = [int(i) for i in cov_idx] if cov_idx else []
        feat_idx = target_idx + cov_idx
        seen = set()
        feat_idx = [i for i in feat_idx if not (i in seen or seen.add(i))]

        path = f"{root}/Data/{dataset}/{parquet}.npy"
        large = self._get_memmap(path)
        seq_sel = large[start:stop]
        if seq_sel.ndim == 1:
            seq_sel = seq_sel.reshape(-1, 1)
        seq_sel = seq_sel[:, feat_idx] if len(feat_idx) > 0 else seq_sel[:, :0]

        full_length = seq_sel.shape[0]
        pred_len = self.pred_len
        min_past = pred_len
        if full_length >= 128:
            slice_idx = full_length - pred_len
            if training:
                max_begin = full_length - pred_len - min_past
                begin_idx = np.random.randint(0, max_begin + 1) if max_begin > 0 else 0
            else:
                begin_idx = 0
        else:
            begin_idx = 0
            slice_idx = (full_length + 1) // 2

        if slice_idx - begin_idx >= self.block_size:
            context = seq_sel[slice_idx - self.block_size: slice_idx]
        else:
            context = seq_sel[begin_idx: slice_idx]

        future = seq_sel[slice_idx: full_length]
        if future.shape[0] > pred_len:
            future = future[:pred_len]
        future_observed = self._observed_mask(future)
        future_raw = np.where(future_observed, future, np.nan)

        combined = np.concatenate([context, future], axis=0)
        _, mean, std = self.normalize_with_stats(combined, context_length=context.shape[0])

        y_raw = np.zeros_like(y)
        k = min(len(feat_idx), y.shape[-1])
        if k > 0 and future.shape[0] > 0:
            future_take = min(future.shape[0], pred_len)
            y_raw[:future_take, :k] = np.nan_to_num(future_raw[:future_take, :k], nan=0.0).astype(np.float32)

        return x, y, target_mask, mean, std, y_raw

    def make_dataset(
        self,
        data,
        batch_size=256,
        training: bool = False,
        seed: int = 0,
        shuffle_buffer_size: int = 1_000_000,
        shuffle=None,
    ):
        if shuffle is None:
            shuffle = training
        if int(shuffle_buffer_size) <= 0:
            shuffle = False
        dataset = _IndexIterableDataset(
            data=data,
            data_handler=self,
            training=training,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
            shuffle=bool(shuffle),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )


class _IndexIterableDataset(IterableDataset):
    def __init__(
        self,
        data,
        data_handler: DataHandler,
        training: bool,
        seed: int,
        shuffle_buffer_size: int,
        shuffle: bool,
    ):
        super().__init__()
        self.data = data
        self.data_handler = data_handler
        self.training = training
        self.seed = int(seed)
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.shuffle = bool(shuffle)

    def __iter__(self):
        rng = random.Random(self.seed)
        total_sequences = len(self.data)
        if total_sequences == 0:
            return
        while True:
            if self.shuffle:
                indices = range(total_sequences)
                buffer_size = min(total_sequences, self.shuffle_buffer_size)
                iterator = iter(indices)
                buffer = []
                for _ in range(buffer_size):
                    try:
                        buffer.append(next(iterator))
                    except StopIteration:
                        break
                while buffer:
                    j = rng.randrange(len(buffer))
                    idx = buffer[j]
                    try:
                        buffer[j] = next(iterator)
                    except StopIteration:
                        buffer.pop(j)
                    yield self._load(idx)
            else:
                for idx in range(total_sequences):
                    yield self._load(idx)

    def _load(self, idx):
        x, y, target_mask = self.data_handler.index_to_array(self.data[int(idx)], training=self.training)
        y_pack = np.concatenate([y, target_mask], axis=-1).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y_pack)
