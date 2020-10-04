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
    def __init__(self):
        super(PostProcessingNetConv1d, self).__init__()
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
        emb_fusion = torch.cat((emb_sample.unsqueeze(1), emb_task.unsqueeze(1)), dim=1) # (bs, 2, d/2)
        out = self.layer1(emb_fusion)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class PostProcessingSelfAttnModule(nn.Module):
    def __init__(self, in_dim, ratio=4):
        """
        Input size is (bs, in_dim, L).
        Should consider 1d Conv.
        """
        super(PostProcessingSelfAttnModule, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//ratio , kernel_size= 1)
        self.key_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim//ratio , kernel_size= 1)
        self.value_conv = nn.Conv1d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X L)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        batchsize, C, L = x.size()
        proj_query = self.query_conv(x).permute(0,2,1) # .view(batchsize, -1, width*height).permute(0,2,1) # B X N X C
        proj_key = self.key_conv(x) # .view(batchsize, -1, width*height) # B x C x N
        energy = torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # B X N X N
        proj_value = self.value_conv(x).view(batchsize,-1,L) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(batchsize,C,L)

        out = self.gamma * out + x
        return out, attention

class PostProcessingNetConv1d_SelfAttn(nn.Module):
    # def __init__(self, out_dim=None, hidden_dim=None, dataset='MiniImageNet', task_embedding='None'):
    def __init__(self, dataset='miniImageNet'):
        super(PostProcessingNetConv1d_SelfAttn, self).__init__()
        self.dataset = dataset
        # self.skip_attn1 = skip_attn1
        if self.dataset == 'miniImageNet' or self.dataset == 'tieredImageNet':
            self.out_channels = [16, 64, 128]
            self.max_pool_ks = [8, 4]
            self.ratio = 8
            self.skip_attn1 = True
        else:
            self.out_channels = [4, 8, 16]
            self.max_pool_ks = [2, 2]
            self.ratio = 4
            self.skip_attn1 = False

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=self.out_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.out_channels[0]),
            nn.ReLU(),
            nn.MaxPool1d(self.max_pool_ks[0])
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=self.out_channels[0], out_channels=self.out_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(self.out_channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(self.max_pool_ks[1])
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=self.out_channels[1], out_channels=self.out_channels[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(self.out_channels[2])
        )
        # self.layer1 = nn.Sequential(
        #     nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(4),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2)
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm1d(8),
        #     nn.ReLU(),
        #     nn.MaxPool1d(2)
        # )
        # self.layer3 = nn.Sequential(
        #     nn.Conv1d(in_channels=8, out_channels=16, kernel_size=4, stride=2, padding=1),
        #     nn.BatchNorm1d(16)
        # )

        self.attn1 = PostProcessingSelfAttnModule(in_dim=self.out_channels[0], ratio=self.ratio)
        self.attn2 = PostProcessingSelfAttnModule(in_dim=self.out_channels[1], ratio=self.ratio)

    def forward(self, x):
        emb_sample, emb_task = x.split(x.size(1) // 2, dim=1) # (bs, d/2), (bs, d/2)
        emb_fusion = torch.cat((emb_sample.unsqueeze(1), emb_task.unsqueeze(1)), dim=1) # (bs, 2, d/2)
        out = self.layer1(emb_fusion)
        if not self.skip_attn1:
            out, _ = self.attn1(out)
        out = self.layer2(out)
        out, _ = self.attn2(out)
        out = self.layer3(out)
        return out


class Identity(nn.Module):
    r"""A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])

    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input
