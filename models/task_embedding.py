import os
import sys
import ipdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable

from models.relation_net import Relation
from models.classification_heads import ClassificationHead


def computeGradientPenalty(outputs, inputs):
    return autograd.grad(
        outputs=outputs, inputs=inputs, grad_outputs=torch.ones_like(outputs),
        create_graph=False, retain_graph=False, only_inputs=True)[0]


def TaskEmbedding_None(emb_support, emb_query, *args):
    # NOTE: `None` in the return statement is preserved for G
    return emb_support, emb_query, None, None


def TaskEmbedding_KME(emb_support, emb_query, *args):
    emb_task = emb_support.mean(dim=1, keepdim=True)
    augmented_support = torch.cat([emb_support, emb_task.expand_as(emb_support)], dim=-1)
    augmented_query = torch.cat([emb_query, emb_task.expand_as(emb_query)], dim=-1)
    # NOTE: `None` in the return statement is preserved for G
    return augmented_support, augmented_query, None, None


def TaskEmbedding_FiLM_KME(emb_support, *args):
    emb_task = emb_support.mean(dim=1, keepdim=True)
    return emb_task, None


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


class TaskEmbedding_FiLM_Entropy_SVM(nn.Module):
    def __init__(self):
        super(TaskEmbedding_FiLM_Entropy_SVM, self).__init__()
        self.cls_head = ClassificationHead(base_learner='SVM-CS')

    def forward(self, emb_support, labels_support, train_way, train_shot):
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
        return emb_task


class TaskEmbedding_Cat_SVM_WGrad(nn.Module):
    def __init__(self):
        super(TaskEmbedding_Cat_SVM_WGrad, self).__init__()
        self.cls_head = ClassificationHead(base_learner='SVM-CS-WNorm')

    def forward(self, emb_support, emb_query, data_support, data_query,
                labels_support, train_way, train_shot, prune_ratio=0.0):
        n_episode, n_support, d = emb_support.size()
        # Train the SVM head
        logit_support, wnorm = self.cls_head(emb_support, emb_support, labels_support, train_way, train_shot)

        # Compute the gradient of `wnorm` w.r.t. `emb_support`
        wgrad = computeGradientPenalty(wnorm, emb_support) # (tasks_per_batch, n_support, d)
        wgrad_abs = wgrad.abs()
        # Normalize gradient
        with torch.no_grad():
            # Prune the gradient according to the magnitude
            if prune_ratio > 0:
                assert prune_ratio < 1.0
                num_pruned = int(d * prune_ratio)
                threshold = torch.kthvalue(wgrad_abs, k=num_pruned, dim=-1, keepdim=True)[0].detach()
                wgrad_abs[wgrad_abs <= threshold] = 0.0
            wgrad_abs_sum = torch.sum(wgrad_abs, dim=(1,2), keepdim=True).detach()
        G = wgrad_abs / wgrad_abs_sum * d

        # Compute task features
        emb_task = (emb_support * G).sum(dim=1, keepdim=True) # (tasks_per_batch, 1, d)

        augmented_support = torch.cat([emb_support, emb_task.expand_as(emb_support)], dim=-1)
        augmented_query = torch.cat([emb_query, emb_task.expand_as(emb_query)], dim=-1)
        # NOTE: `None` in the return statement is preserved for G
        return augmented_support, augmented_query, None, None


class TaskEmbedding_FiLM_SVM_WGrad(nn.Module):
    def __init__(self):
        super(TaskEmbedding_FiLM_SVM_WGrad, self).__init__()
        self.cls_head = ClassificationHead(base_learner='SVM-CS-WNorm')

    def forward(self, emb_support, labels_support, train_way, train_shot,
                prune_ratio=0.0):
        n_episode, n_support, d = emb_support.size()
        # Train the SVM head
        logit_support, wnorm = self.cls_head(
            emb_support, emb_support,labels_support, train_way, train_shot)

        # Compute the gradient of `wnorm` w.r.t. `emb_support`
        wgrad = computeGradientPenalty(wnorm, emb_support)
        # wgrad -> (tasks_per_batch, n_support, d)
        wgrad_abs = wgrad.abs()
        # Normalize gradient
        with torch.no_grad():
            # Prune the gradient according to the magnitude
            if prune_ratio > 0:
                assert prune_ratio < 1.0
                num_pruned = int(d * prune_ratio)
                threshold = torch.kthvalue(
                    wgrad_abs, k=num_pruned, dim=-1, keepdim=True)[0].detach()
                wgrad_abs[wgrad_abs <= threshold] = 0.0
            wgrad_abs_sum = torch.sum(wgrad_abs, dim=(1,2), keepdim=True)
        G = wgrad_abs / wgrad_abs_sum * d

        # Compute task features
        emb_task = (emb_support * G).sum(dim=1, keepdim=True)

        return emb_task, wgrad


class TaskEmbedding_FiLM_SVM_OnW(nn.Module):
    def __init__(self):
        super(TaskEmbedding_FiLM_SVM_OnW, self).__init__()
        self.cls_head = ClassificationHead(base_learner='SVM-CS-OnW')

    def forward(self, emb_support, labels_support, train_way, train_shot,
                prune_ratio=0.0):
        n_episode, n_support, d = emb_support.size()
        # Train the SVM head
        logit_support, w = self.cls_head(
            emb_support, emb_support,labels_support, train_way, train_shot)

        assert len(w.shape) == 2
        # # Compute the gradient of `wnorm` w.r.t. `emb_support`

        return w.unsqueeze(1), None


class TaskEmbedding_Entropy_RidgeHead(nn.Module):
    def __init__(self):
        super(TaskEmbedding_Entropy_RidgeHead, self).__init__()
        self.cls_head = ClassificationHead(base_learner='Ridge')

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


class TaskEmbedding_Entropy_SVMHead_NoGrad(nn.Module):
    def __init__(self):
        super(TaskEmbedding_Entropy_SVMHead_NoGrad, self).__init__()
        self.cls_head = ClassificationHead(base_learner='SVM-CS')

    def forward(self, emb_support, emb_query, data_support, data_query,
                labels_support, train_way, train_shot):
        with torch.no_grad():
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
        # NOTE: Because the authors of this chunk of codes used `in` keyword
        #       to check the option for `metric`, please make sure that the
        #       name of former checked name is not a sub-string of the metric 
        #       names that are checked later.
        if ('KME' == metric):
            self.te_func = TaskEmbedding_KME
        elif ('FiLM_KME' == metric):
            self.te_func = TaskEmbedding_FiLM_KME
        elif ('Cosine' in metric):
            self.te_func = TaskEmbedding_Cosine
        elif ('Entropy_SVM_NoGrad' in metric):
            self.te_func = TaskEmbedding_Entropy_SVMHead_NoGrad()
        elif ('Entropy_SVM' in metric):
            self.te_func = TaskEmbedding_Entropy_SVMHead()
        elif ('FiLM_Entropy_SVM' in metric):
            self.te_func = TaskEmbedding_FiLM_Entropy_SVM()
        elif ('Cat_SVM_WGrad' in metric):
            self.te_func = TaskEmbedding_Cat_SVM_WGrad()
        elif ('FiLM_SVM_WGrad' in metric):
            self.te_func = TaskEmbedding_FiLM_SVM_WGrad()
        elif ('FiLM_SVM_OnW' in metric):
            self.te_func = TaskEmbedding_FiLM_SVM_OnW()
        elif ('Entropy_Ridge' in metric):
            self.te_func = TaskEmbedding_Entropy_RidgeHead()
        elif ('Relation' in metric):
            self.te_func = Relation(dataset=dataset)
        elif ('None' in metric):
            self.te_func = TaskEmbedding_None
        else:
            print ("Cannot recognize the metric type {}".format(metric))
            assert(False)

    def forward(self, *args):
        return self.te_func(*args)
