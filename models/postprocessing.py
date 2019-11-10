import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PostProcessingNet(nn.Module):
    def __init__(self, out_dim=None, hidden_dim=None, dataset='MiniImageNet', task_embedding='None'):
        super(PostProcessingNet, self).__init__()
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.dataset = dataset
        self.task_embedding = task_embedding

        if 'imagenet' in self.dataset.lower():
            self.in_dim = 16000
            self.dropout = 0.2
        elif 'cifar' in self.dataset.lower() or 'FS' in self.dataset:
            self.in_dim = 2560
            self.dropout = 0.2
        elif 'omniglot' in self.dataset.lower():
            raise NotImplementedError('Post processing module is not implemented for Omniglot dataset yet')
        else:
            raise ValueError('Cannot recognize dataset {}'.format(self.dataset))

        if self.task_embedding != 'None':
            self.in_dim *= 2

        if self.out_dim is None:
            self.out_dim = self.in_dim
        if self.hidden_dim is None:
            self.hidden_dim = self.out_dim

        self.layer1 = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout))
        self.layer2 = nn.Linear(self.hidden_dim, self.out_dim)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

