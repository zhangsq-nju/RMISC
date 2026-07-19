import math

import torch
from torch import nn
import torch.nn.functional as F


class TSPositionalEncoding(nn.Module):
    def __init__(self, num_positions: int, embed_dim: int):
        super().__init__()
        position = torch.arange(0, num_positions, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / embed_dim))
        sin_t = torch.sin(position * div_term)
        cos_t = torch.cos(position * div_term)
        parts = []
        for i in range(sin_t.shape[1]):
            parts.append(sin_t[:, i])
            parts.append(cos_t[:, i])
        pe = torch.stack(parts, dim=1).unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return self.pe[:, : x.shape[1]].to(dtype=x.dtype, device=x.device)


class TSConvPatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, block_size=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.patch_size = patch_size
        self.block_size = block_size
        self.n_embd = embed_dim

    def forward(self, x):
        x = x.unsqueeze(-1)
        t = x.shape[1]
        if t < self.patch_size:
            raise ValueError(f"input length must be at least {self.patch_size}!")
        if self.block_size > t:
            x = F.pad(x, (0, 0, self.block_size - t, 0))
        x = x.transpose(1, 2)
        x = self.conv1(x)
        return x.transpose(1, 2)

