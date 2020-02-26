import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FiLM_Layer(nn.Module):
    def __init__(self, channels, in_channels=1, alpha=1, activation=F.leaky_relu):
        '''
        input size: (N, in_channels). output size: (N, channels)

        Args:
            channels: int.
            alpha: scalar. Expand ratio for FiLM hidden layer.
        '''
        super(FiLM_Layer, self).__init__()
        self.channels = channels
        self.activation = activation
        self.MLP = nn.Sequential(
            nn.Linear(int(in_channels), int(alpha*channels*2), bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(int(alpha*channels*2), int(channels*2), bias=True),
        )

    def forward(self, _input, _lambda):
        N, C, H, W = _input.size()
        if _lambda is not None:
            out = self.MLP(_lambda)
            mu, sigma = torch.split(out, [self.channels, self.channels], dim=-1)
            if self.activation is not None:
                mu, sigma = self.activation(mu), self.activation(sigma)
            mu = mu.view(N, C, 1, 1).expand_as(_input)
            sigma = sigma.view(N, C, 1, 1).expand_as(_input)
        else:
            mu, sigma = torch.ones_like(_input), torch.zeros_like(_input)

        return _input * mu + sigma

