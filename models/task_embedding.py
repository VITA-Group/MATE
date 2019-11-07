import os
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def TaskEmbedding_None(query, key):
    # NOTE: `None` in the return statement is preserved for G
    return query, None

def TaskEmbedding_KME(query, key):
    emb_task = key.mean(dim=1, keepdim=True)
    augmented_query = torch.cat([query, emb_task.expand_as(query)], dim=-1)
    # NOTE: `None` in the return statement is preserved for G
    return augmented_query, None

def TaskEmbedding_Cosine(query, key):
    n_episode, n_query, d = query.size()
    _        , n_key  , _ = key.size()
    expand_query = query.unsqueeze(2).expand(-1, -1, n_key, -1)
    expand_key = key.unsqueeze(1).expand(-1, n_query, -1, -1)
    cosine = F.cosine_similarity(expand_query, expand_key, dim=-1)
    G = F.softmax(cosine, dim=-1)
    emb_task = torch.bmm(G, key)
    augmented_query = torch.cat([query, emb_task], dim=-1)
    return augmented_query, G

def TaskEmbedding_Relation(query, key):
    pass

class TaskEmbedding(nn.Module):
    def __init__(self, metric='None'):
        super(TaskEmbedding, self).__init__()
        if ('KME' in metric):
            self.te_func = TaskEmbedding_KME
        elif ('Cosine' in metric):
            self.te_func = TaskEmbedding_Cosine
        elif ('Relation' in metric):
            raise NotImplementedError('Relation network for task embedding not implemented yet')
        elif ('None' in metric):
            self.te_func = TaskEmbedding_None
        else:
            print ("Cannot recognize the metric type {}".format(metric))
            assert(False)

    def forward(self, query, support):
        return self.te_func(query, support)
