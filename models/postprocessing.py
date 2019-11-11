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


class PostProcessingNetConv1d(nn.Module):
    # def __init__(self, out_dim=None, hidden_dim=None, dataset='MiniImageNet', task_embedding='None'):
    def __init__(self):
        super(PostProcessingNetConv1d, self).__init__()
        # self.out_dim = out_dim
        # self.hidden_dim = hidden_dim
        # self.dataset = dataset
        # self.task_embedding = task_embedding

        # if 'imagenet' in self.dataset.lower():
        #     self.in_dim = 16000
        #     self.dropout = 0.2
        # elif 'cifar' in self.dataset.lower() or 'FS' in self.dataset:
        #     self.in_dim = 2560
        #     self.dropout = 0.2
        # elif 'omniglot' in self.dataset.lower():
        #     raise NotImplementedError('Post processing module is not implemented for Omniglot dataset yet')
        # else:
        #     raise ValueError('Cannot recognize dataset {}'.format(self.dataset))

        # if self.task_embedding != 'None':
        #     self.in_dim *= 2

        # if self.out_dim is None:
        #     self.out_dim = self.in_dim
        # if self.hidden_dim is None:
        #     self.hidden_dim = self.out_dim

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16)
        )

    def forward(self, x):
        emb_sample, emb_task = x.split(x.size(1) // 2, dim=1) # (bs, d/2), (bs, d/2)
        emb_fusion = torch.concat((emb_sample.unsqueeze(1), emb_task.unsqueeze(1)), dim=1) # (bs, 2, d/2)
        out = self.layer1(emb_fusion)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

