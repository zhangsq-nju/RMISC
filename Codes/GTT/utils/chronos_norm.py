import numpy as np


DEFAULT_INVALID_ABS_LIMIT = 1e19


def observed_mask(data, invalid_abs_limit=DEFAULT_INVALID_ABS_LIMIT):
    data = np.asarray(data)
    return np.isfinite(data) & (np.abs(data) < float(invalid_abs_limit))


def normalize_with_stats(
    data,
    context_length=None,
    epsilon=1e-5,
    use_arcsinh=True,
    invalid_abs_limit=DEFAULT_INVALID_ABS_LIMIT,
):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    ref = data[:context_length, :] if context_length is not None else data
    if ref.shape[0] == 0:
        ref = data

    ref_mask = observed_mask(ref, invalid_abs_limit=invalid_abs_limit)
    ref_count = ref_mask.sum(axis=0)
    ref_sum = np.where(ref_mask, ref, 0.0).sum(axis=0, dtype=np.float64)
    loc = np.divide(ref_sum, ref_count, out=np.zeros_like(ref_sum, dtype=np.float64), where=ref_count > 0)
    loc = loc.astype(np.float32, copy=False)

    centered_ref = np.where(ref_mask, ref - loc, 0.0)
    scale_sum = np.square(centered_ref).sum(axis=0, dtype=np.float64)
    scale = np.sqrt(
        np.divide(scale_sum, ref_count, out=np.ones_like(scale_sum, dtype=np.float64), where=ref_count > 0)
    )
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0).astype(np.float32, copy=False)
    scale = np.where(scale == 0, float(epsilon), scale).astype(np.float32, copy=False)

    valid = observed_mask(data, invalid_abs_limit=invalid_abs_limit)
    normalized = (data - loc) / scale
    normalized = np.where(valid, normalized, 0.0)

    if use_arcsinh:
        normalized = np.arcsinh(normalized)
    normalized = np.clip(normalized, -5, 5)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=5.0, neginf=-5.0)
    return normalized.astype(np.float32, copy=False), loc, scale


def denormalize(data, loc, scale, use_arcsinh=True):
    data = np.asarray(data, dtype=np.float32).copy()
    loc = np.asarray(loc, dtype=np.float32).reshape(-1)
    scale = np.asarray(scale, dtype=np.float32).reshape(-1)
    n = min(data.shape[-1], loc.shape[0], scale.shape[0])
    if n > 0:
        values = data[..., :n]
        if use_arcsinh:
            values = np.sinh(values)
        data[..., :n] = values * scale[:n] + loc[:n]
    return data
