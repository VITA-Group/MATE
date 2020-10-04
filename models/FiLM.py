import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FiLM_Layer(nn.Module):
    def __init__(self, channels, in_channels=1, alpha=1, activation=F.leaky_relu,
                 normalize=True):
        '''
        input size: (N, in_channels). output size: (N, channels)

        Args:
            channels: int.
            alpha: scalar. Expand ratio for FiLM hidden layer.
        '''
        super(FiLM_Layer, self).__init__()
        self.channels = channels
        self.activation = activation
        self.mu_multiplier = 1.0
        self.sigma_multiplier = 1.0
        self.normalize = normalize
        self.MLP = nn.Sequential(
            nn.Linear(int(in_channels), int(alpha*channels*2), bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(int(alpha*channels*2), int(channels*2), bias=True),
        )

    def get_mlp_output(self, _task_emb):
        assert _task_emb is not None
        _task_emb = _task_emb.squeeze(1)
        _out = self.MLP(_task_emb)
        if self.normalize:
            denom = _out.abs().sum(dim=1, keepdim=True)
            _out = _out / denom * _out.size(1) * 0.5
        return _out.unsqueeze(1)

    def forward(self, _input, _task_emb, n_expand):
        self._task_emb = _task_emb
        N, C, H, W = _input.size()
        if _task_emb is not None:
            _task_emb = _task_emb.squeeze(1)
            _out = self.MLP(_task_emb)
            if self.normalize:
                denom = _out.abs().sum(dim=1, keepdim=True)
                _out = _out / denom * _out.size(1) * 0.5
            self._out = _out.unsqueeze(1)
            _out = self._out.expand(-1, n_expand, -1).reshape(-1, self._out.size(-1))

            mu, sigma = torch.split(
                _out, [self.channels, self.channels], dim=-1)
            if self.activation is not None:
                mu, sigma = self.activation(mu), self.activation(sigma)

            mu = mu.view(N, C, 1, 1).expand_as(_input) * self.mu_multiplier
            mu = mu.clamp(-1.0, 1.0)
            sigma = sigma.view(N, C, 1, 1).expand_as(_input) * self.sigma_multiplier

        else:
            mu, sigma = torch.zeros_like(_input), torch.zeros_like(_input)

        return _input * (1.0 + mu) + sigma
