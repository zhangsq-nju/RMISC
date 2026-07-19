import torch
from torch import nn

from core.encoder import TSEncoder
from core.revin import RevIN


class GTTNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.block_size = config.block_size
        self.target_dim = config.target_dim
        self.covariate_dim = config.covariate_dim
        self.timefeat_dim = config.timefeat_dim
        self.n_embd = config.n_embd
        self.forecast_mode = config.forecast_mode
        self.enable_revin = config.enable_revin
        self.revin_time = config.revin_time
        self.pred_len = config.pred_len

        if self.enable_revin:
            self.revin = RevIN(
                num_features=self.target_dim + self.covariate_dim + self.timefeat_dim,
                affine=config.affine,
            )
        self.encoder = TSEncoder(config)
        self.mu_head = nn.Linear(config.n_embd, self.patch_size if self.pred_len is None else self.pred_len)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs):
        if self.enable_revin:
            if self.timefeat_dim > 0 and self.revin_time is False:
                x_val = inputs[:, :, : self.target_dim + self.covariate_dim]
                x_time = inputs[:, :, -self.timefeat_dim :]
                x_val = self.revin(x_val, mode="norm")
                x_enc = torch.cat([x_val, x_time], dim=-1)
            else:
                x_enc = self.revin(inputs, mode="norm")
        else:
            x_enc = inputs

        _, t, c = x_enc.shape
        patch_num = self.block_size // self.patch_size
        x_enc = x_enc.transpose(1, 2).reshape(-1, t)
        x_dec = self.encoder(x_enc, patch_num=patch_num, channel_num=c)
        x_dec = x_dec[:, -1, :]
        outputs_mu = self.mu_head(x_dec)
        if self.pred_len is None:
            outputs_mu = outputs_mu.reshape(-1, c, self.patch_size)
        else:
            outputs_mu = outputs_mu.reshape(-1, c, self.pred_len)
        outputs_mu = outputs_mu.transpose(1, 2)
        if self.revin_time is False:
            outputs_mu = outputs_mu[:, :, : self.target_dim + self.covariate_dim]
        if self.enable_revin:
            outputs_mu = self.revin(outputs_mu, mode="denorm")
        return outputs_mu[:, :, : self.target_dim]

    @classmethod
    def build_raw_model(cls, mc):
        model = cls(mc)
        input_dim = mc.target_dim + mc.covariate_dim + mc.timefeat_dim
        with torch.no_grad():
            model(torch.zeros((1, mc.block_size, input_dim), dtype=torch.float32))
        return model

