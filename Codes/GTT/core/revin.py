import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible Instance Normalization for Accurate Time-Series Forecasting."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        self.mean = None
        self.stdev = None

    def forward(self, inputs, mode):
        if mode == "norm":
            self._get_statistics(inputs)
            return self._normalize(inputs)
        if mode == "denorm":
            return self._denormalize(inputs)
        raise NotImplementedError("Only modes norm and denorm are supported.")

    def _get_statistics(self, x):
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = (x.var(dim=1, keepdim=True, unbiased=False).sqrt() + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * self.stdev + self.mean

