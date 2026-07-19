import torch
from torch import nn
import torch.nn.functional as F

from core.embedding import TSConvPatchEmbedding, TSPositionalEncoding


def get_max_embd_length(config):
    return config.block_size // config.patch_size


class TSAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_mask_len: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got {embed_dim} and {num_heads}).")
        self.scaling = self.head_dim ** -0.5
        self.max_mask_len = max_mask_len
        mask = torch.tril(torch.ones(max_mask_len, max_mask_len)).view(1, 1, max_mask_len, max_mask_len)
        self.register_buffer("mask", mask, persistent=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int):
        return tensor.reshape(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, hidden_states, enable_masking=True):
        _, tgt_len, embed_dim = hidden_states.shape
        src_len = tgt_len
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self._shape(self.k_proj(hidden_states), tgt_len)
        value_states = self._shape(self.v_proj(hidden_states), tgt_len)

        key_states = key_states.reshape(-1, src_len, self.head_dim).float()
        value_states = value_states.reshape(-1, src_len, self.head_dim)
        query_states = self._shape(query_states, tgt_len).reshape(-1, tgt_len, self.head_dim).float()

        attn_weights = torch.matmul(query_states, key_states.transpose(1, 2))
        if enable_masking:
            attn_weights = attn_weights.reshape(-1, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                self.mask[:, :, :tgt_len, :src_len].to(device=attn_weights.device) == 0,
                float("-inf"),
            )
            attn_weights = attn_weights.reshape(-1, tgt_len, src_len)

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_probs.to(value_states.dtype), value_states)
        attn_output = attn_output.reshape(-1, self.num_heads, tgt_len, self.head_dim).transpose(1, 2)
        attn_output = attn_output.reshape(-1, tgt_len, embed_dim)
        return self.out_proj(attn_output)


class TSEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.n_embd
        self.self_attn1 = TSAttention(
            self.embed_dim,
            config.encoder_attention_heads,
            max_mask_len=get_max_embd_length(config),
            dropout=config.attention_dropout,
        )
        self.self_attn2 = TSAttention(
            self.embed_dim,
            config.encoder_attention_heads,
            max_mask_len=get_max_embd_length(config),
            dropout=config.attention_dropout,
        )
        self.temporal_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.channel_attn_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.dropout = nn.Dropout(config.dropout)
        self.activation_dropout = nn.Dropout(config.activation_dropout)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)

    def forward(self, hidden_states, patch_num, channel_num):
        residual = hidden_states
        hidden_states = self.temporal_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn1(hidden_states=hidden_states, enable_masking=False)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = hidden_states.reshape(-1, channel_num, patch_num, self.embed_dim)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(-1, channel_num, self.embed_dim)
        hidden_states = self.channel_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn2(hidden_states=hidden_states, enable_masking=False)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states.reshape(-1, patch_num, channel_num, self.embed_dim)
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.reshape(-1, patch_num, self.embed_dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = F.gelu(self.fc1(hidden_states))
        hidden_states = self.activation_dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return residual + hidden_states


class TSEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layerdrop = config.encoder_layerdrop
        self.embed_dim = config.n_embd
        self.max_embed_length = get_max_embd_length(config)
        self.position_embedding = TSPositionalEncoding(self.max_embed_length, embed_dim=config.n_embd)
        self.patch_embedding = TSConvPatchEmbedding(config.patch_size, config.n_embd, config.block_size)
        self.dropout = nn.Dropout(config.embedd_pdrop)
        self.encoder_layers = nn.ModuleList([TSEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-5)

    def forward(self, x, patch_num, channel_num):
        hidden_states = self.patch_embedding(x)
        if patch_num <= self.max_embed_length:
            hidden_states = hidden_states.float() + self.position_embedding(hidden_states)
        else:
            raise ValueError("patch_num must be less than 16!")
        hidden_states = self.dropout(hidden_states)
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states, patch_num=patch_num, channel_num=channel_num)
        return self.layer_norm(hidden_states)

