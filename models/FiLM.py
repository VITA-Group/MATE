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
        self.mu_multiplier = 1.0
        self.sigma_multiplier = 1.0
        # self.mu_multiplier = nn.Parameter(torch.ones(1).float())
        # self.sigma_multiplier = nn.Parameter(torch.ones(1).float())
        # self.mu_multiplier = nn.Parameter(torch.zeros(1).float())
        # self.sigma_multiplier = nn.Parameter(torch.zeros(1).float())
        self.MLP = nn.Sequential(
            nn.Linear(int(in_channels), int(alpha*channels*2), bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(int(alpha*channels*2), int(channels*2), bias=True),
        )
        # self.MLP = nn.Linear(int(in_channels), int(alpha*channels*2), bias=True)

    def forward(self, _input, _lambda):
        # print(_input.abs().mean())
        N, C, H, W = _input.size()
        if _lambda is not None:
            out = self.MLP(_lambda)
            mu, sigma = torch.split(out, [self.channels, self.channels], dim=-1)
            if self.activation is not None:
                mu, sigma = self.activation(mu), self.activation(sigma)
            mu = mu.view(N, C, 1, 1).expand_as(_input) * self.mu_multiplier
            mu = mu.clamp(-1.0, 1.0)
            sigma = sigma.view(N, C, 1, 1).expand_as(_input) * self.sigma_multiplier
            # print(_input.abs().mean().cpu().item(), mu.abs().mean().cpu().item(), sigma.abs().mean().cpu().item())
            # print(mu.size(), sigma.size())
            # print(mu[0,:,0,0], sigma[0,:,0,0])
            # print(self.mu_multiplier, self.sigma_multiplier)

        else:
            mu, sigma = torch.zeros_like(_input), torch.zeros_like(_input)

        return _input * (1.0 + mu) + sigma
        # return _input * (1.0 + mu) + sigma

