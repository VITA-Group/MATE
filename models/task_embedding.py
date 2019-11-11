import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.relation_net import Relation
from models.classification_heads import ClassificationHead


def TaskEmbedding_None(emb_support, emb_query, *args):
    # NOTE: `None` in the return statement is preserved for G
    return emb_support, emb_query, None, None


def TaskEmbedding_KME(emb_support, emb_query, *args):
    emb_task = emb_support.mean(dim=1, keepdim=True)
    augmented_support = torch.cat([emb_support, emb_task.expand_as(emb_support)], dim=-1)
    augmented_query = torch.cat([emb_query, emb_task.expand_as(emb_query)], dim=-1)
    # NOTE: `None` in the return statement is preserved for G
    return augmented_support, augmented_query, None, None


def TaskEmbedding_Cosine(emb_support, emb_query, *args):
    n_support = emb_support.size()[1]
    n_query   = emb_query.size()[1]
    expand_support = emb_support.unsqueeze(2).expand(-1, -1, n_support, -1)
    expand_query   = emb_query.unsqueeze(2).expand(-1, -1, n_support, -1)
    expand_support_key = emb_support.unsqueeze(1).expand(-1, n_support, -1, -1)
    expand_query_key   = emb_support.unsqueeze(1).expand(-1, n_query, -1, -1)
    cosine_support = F.cosine_similarity(expand_support, expand_support_key, dim=-1)
    cosine_query   = F.cosine_similarity(expand_query  , expand_query_key  , dim=-1)
    G_support = F.softmax(cosine_support, dim=-1)
    G_query   = F.softmax(cosine_query  , dim=-1)
    emb_task_support = torch.bmm(G_support, emb_support)
    emb_task_query   = torch.bmm(G_query  , emb_support)
    augmented_support = torch.cat([emb_support, emb_task_support], dim=-1)
    augmented_query   = torch.cat([emb_query  , emb_task_query]  , dim=-1)
    return augmented_support, augmented_query, G_support, G_query


class TaskEmbedding_Entropy_SVMHead(nn.Module):
    def __init__(self):
        super(TaskEmbedding_Entropy_SVMHead, self).__init__()
        self.cls_head = ClassificationHead(base_learner='SVM-CS')

    def forward(self, emb_support, emb_query, data_support, data_query,
                labels_support, train_way, train_shot):
        n_episode, n_support = emb_support.size()[:2]
        logit_support = self.cls_head(emb_support, emb_support, labels_support, train_way, train_shot)
        logit_support_rsp = logit_support.reshape(n_episode * n_support, train_way)
        prb = F.softmax(logit_support_rsp, dim=1)
        log_prb = F.log_softmax(logit_support_rsp, dim=1)
        entropy = - (prb * log_prb).sum(dim=1).reshape(n_episode, n_support, 1)
        G = np.log(train_way) - entropy
        # normalize G
        G = G / G.sum(dim=1,keepdim=True)
        emb_task = (emb_support * G).mean(dim=1, keepdim=True)
        augmented_support = torch.cat([emb_support, emb_task.expand_as(emb_support)], dim=-1)
        augmented_query = torch.cat([emb_query, emb_task.expand_as(emb_query)], dim=-1)
        # NOTE: `None` in the return statement is preserved for G
        return augmented_support, augmented_query, entropy, entropy

class TaskEmbedding(nn.Module):
    def __init__(self, metric='None', dataset='MiniImageNet'):
        super(TaskEmbedding, self).__init__()
        if ('KME' in metric):
            self.te_func = TaskEmbedding_KME
        elif ('Cosine' in metric):
            self.te_func = TaskEmbedding_Cosine
        elif ('Entropy_SVM' in metric):
            self.te_func = TaskEmbedding_Entropy_SVMHead()
        elif ('Relation' in metric):
            self.te_func = Relation(dataset=dataset)
        elif ('None' in metric):
            self.te_func = TaskEmbedding_None
        else:
            print ("Cannot recognize the metric type {}".format(metric))
            assert(False)

    def forward(self, emb_support, emb_query, data_support, data_query,
                labels_support, train_way, train_shot):
        return self.te_func(
            emb_support, emb_query, data_support, data_query,
            labels_support, train_way, train_shot)
