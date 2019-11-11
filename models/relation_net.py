import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationNetworkEncoder(nn.Module):
    """ RelationNetworkEncoder

    We use a universal encoder for all of the three datasets --- miniimagenet,
    cifarfs and ominiglot.

    Input shape:
        miniimagenet: (BS, 3, 84, 84)
        cifarfs     : (BS, 3, 32, 32)
        omniglot    : (BS, 3, 28, 28)

    Output shape:
        miniimagenet: (BS, 64, 19, 19)
        cifarfs     : (BS, 64, 6, 6)
        omniglot    : (BS, 64, 5, 5)
    """
    def __init__(self):
        super(RelationNetworkEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, hidden_size, dataset='miniimagenet'):
        super(RelationNetwork, self).__init__()
        self.dataset = dataset
        if 'imagenet' in self.dataset.lower():
            self.paddings = [0, 0]
            self.feature_size = 64 * 3 * 3
        elif 'cifar' in self.dataset.lower() or 'FS' in self.dataset:
            self.paddings = [0, 1]
            self.feature_size = 64
        elif 'omniglot' in self.dataset.lower():
            self.paddings = [1, 1]
            self.feature_size = 64
        else:
            raise ValueError('Cannot recognize dataset {}'.format(self.dataset))

        self.layer1 = nn.Sequential(
            nn.Conv2d(64*2,64,kernel_size=3,padding=self.paddings[0]),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,padding=self.paddings[1]),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(self.feature_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


class Relation(nn.Module):
    """docstring for RelationNetwork"""
    def __init__(self, hidden_size=64, dataset='miniimagenet'):
        super(Relation, self).__init__()
        self.dataset = dataset
        self.hidden_size = hidden_size
        self.encoder = RelationNetworkEncoder()
        self.relation_network = RelationNetwork(hidden_size, dataset)

    def forward(self, emb_support, emb_query, data_support, data_query, *args):
        """
        The input is an episode of samples [xs, xq] of shape (n_support + n_query, im_w, im_h).

        First, all samples are passed through the encoder.
        Second, construct the attention matrix G of shape (n_support + n_query, n_support)
        """
        n_episode, n_support = data_support.size()[:2]
        n_query = data_query.size()[1]

        data_support = self.encoder(data_support.reshape([-1] + list(data_support.size()[2:])))
        data_support = data_support.reshape([n_episode, n_support] + list(data_support.size()[1:]))

        data_query = self.encoder(data_query.reshape([-1] + list(data_query.size()[2:])))
        data_query = data_query.reshape([n_episode, n_query] + list(data_query.size()[1:]))

        expand_data_support = data_support.unsqueeze(2).expand(-1, -1, n_support, *data_support.size()[2:])
        expand_data_query = data_query.unsqueeze(2).expand(-1, -1, n_support, *data_query.size()[2:])
        data_key = data_support.unsqueeze(1)

        relation_input_support = torch.cat([expand_data_support, data_key.expand_as(expand_data_support)], dim=3)
        relation_input_query = torch.cat([expand_data_query, data_key.expand_as(expand_data_query)], dim=3)

        G_support = self.relation_network(relation_input_support.reshape([-1] + list(relation_input_support.size()[3:])))
        G_query = self.relation_network(relation_input_query.reshape([-1] + list(relation_input_query.size()[3:])))

        G_support = G_support.reshape(n_episode, n_support, n_support)
        G_query = G_query.reshape(n_episode, n_query, n_support)

        emb_task_support = torch.bmm(G_support, emb_support)
        emb_task_query   = torch.bmm(G_query  , emb_support)
        augmented_support = torch.cat([emb_support, emb_task_support], dim=-1)
        augmented_query   = torch.cat([emb_query  , emb_task_query]  , dim=-1)

        return augmented_support, augmented_query, G_support, G_query

