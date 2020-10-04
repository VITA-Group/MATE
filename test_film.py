# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.autograd import Variable

from tqdm import tqdm

from models.protonet_embedding import ProtoNetEmbedding
from models.R2D2_embedding import R2D2Embedding
from models.ResNet12_embedding import resnet12
from models.ResNet12_FiLM_embedding import resnet12_film, ResNet_FiLM
from models.resnet_rfs import resnet12_rfs
from models.resnet_rfs_film import resnet12_rfs_film
from models.task_embedding import TaskEmbedding
from models.postprocessing import Identity, PostProcessingNet, PostProcessingNetConv1d, PostProcessingNetConv1d_SelfAttn

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from models.classification_heads import ClassificationHead

from utils import pprint, set_gpu, Timer, count_accuracy, log

import numpy as np
import os


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        if 'imagenet' in options.dataset.lower():
            network = resnet12(avg_pool=False,
                               drop_rate=0.1,
                               dropblock_size=5).cuda()
        else:
            network = resnet12(avg_pool=False,
                               drop_rate=0.1,
                               dropblock_size=2).cuda()
        device_ids = list(range(len(options.gpu.split(','))))
        network = torch.nn.DataParallel(network, device_ids=device_ids)
    elif options.network == 'ResNetRFS':
        if 'imagenet' in opt.dataset.lower():
            network = resnet12_rfs(avg_pool=True,
                                   drop_rate=0.1,
                                   dropblock_size=5).cuda()
        else:
            network = resnet12_rfs(avg_pool=True,
                                   drop_rate=0.1,
                                   dropblock_size=2).cuda()
        device_ids = list(range(len(options.gpu.split(','))))
        network = torch.nn.DataParallel(network, device_ids=device_ids)
    elif options.network == 'ResNet_FiLM':
        film_act = None if options.no_film_activation else F.leaky_relu
        if 'imagenet' in options.dataset.lower():
            network = resnet12_film(
                avg_pool=False, drop_rate=0.1, dropblock_size=5,
                film_indim=opt.film_indim, film_alpha=1.0, film_act=film_act,
                final_relu=(not opt.no_final_relu),
                film_normalize=opt.film_normalize,
                dual_BN=options.dual_BN).cuda()
            options.film_preprocess_input_dim = 16000
        else:
            network = resnet12_film(
                avg_pool=False, drop_rate=0.1, dropblock_size=2,
                film_indim=options.film_indim, film_alpha=1.0, film_act=film_act,
                final_relu=(not opt.no_final_relu),
                film_normalize=opt.film_normalize,
                dual_BN=options.dual_BN).cuda()
        device_ids = list(range(len(options.gpu.split(','))))
        network = torch.nn.DataParallel(network, device_ids=device_ids)
    elif options.network == 'ResNetRFS_FiLM':
        film_act = None if options.no_film_activation else F.leaky_relu
        if 'imagenet' in opt.dataset.lower():
            network = resnet12_rfs_film(
                avg_pool=True, drop_rate=0.1, dropblock_size=5,
                film_indim=640, film_alpha=1.0, film_act=film_act,
                final_relu=(not options.no_final_relu),
                film_normalize=options.film_normalize,
                dual_BN=options.dual_BN).cuda()
        else:
            network = resnet12_rfs_film(
                avg_pool=True, drop_rate=0.1, dropblock_size=2,
                film_indim=640, film_alpha=1.0, film_act=film_act,
                final_relu=(not options.no_final_relu),
                film_normalize=options.film_normalize,
                dual_BN=options.dual_BN).cuda()
        device_ids = list(range(len(options.gpu.split(','))))
        network = torch.nn.DataParallel(network, device_ids=device_ids)
    else:
        print("Cannot recognize the network type")
        assert False

    # Choose the classification head
    if opt.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif opt.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif opt.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif opt.head == 'SVM' or opt.head == 'LR':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    elif options.head == 'SVM-BiP':
        cls_head = ClassificationHead(base_learner='SVM-CS-BiP').cuda()
    else:
        print("Cannot recognize the classification head type")
        assert False

    return network, cls_head


def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_test = MiniImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_test = tieredImageNet(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_test = CIFAR_FS(phase='test')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_test = FC100(phase='test')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)

    return (dataset_test, data_loader)


def get_task_embedding_func(options):
    # Choose the task embedding function
    te_args = dict(dataset=options.dataset) if options.task_embedding == 'Relation' else dict()
    te_func = TaskEmbedding(metric=options.task_embedding, **te_args).cuda()

    device_ids = list(range(len(options.gpu.split(','))))
    te_func = torch.nn.DataParallel(te_func, device_ids=device_ids)

    return te_func


def get_postprocessing_model(options):
    # Choose the post processing network for embeddings
    if options.post_processing == 'FC':
        return PostProcessingNet(
            dataset=options.dataset,
            task_embedding=options.task_embedding).cuda()

    if options.post_processing == 'Conv1d':
        postprocessing_net = PostProcessingNetConv1d().cuda()
        device_ids = list(range(len(options.gpu.split(','))))
        postprocessing_net = torch.nn.DataParallel(
            postprocessing_net, device_ids=device_ids)
        return postprocessing_net

    if options.post_processing == 'Conv1d_SelfAttn':
        postprocessing_net = PostProcessingNetConv1d_SelfAttn(
            dataset=options.dataset).cuda()
        device_ids = list(range(len(options.gpu.split(','))))
        postprocessing_net = torch.nn.DataParallel(
            postprocessing_net, device_ids=device_ids)
        return postprocessing_net

    elif options.post_processing == 'None':
        return Identity().cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./experiments/exp_1/best_model.pth',
                        help='path of the checkpoint file')
    parser.add_argument('--episode', type=int, default=1000,
                        help='number of episodes to test')
    parser.add_argument('--way', type=int, default=5,
                        help='number of classes in one test episode')
    parser.add_argument('--shot', type=int, default=1,
                        help='number of support examples per training class')
    parser.add_argument('--query', type=int, default=15,
                        help='number of query examples per training class')
    parser.add_argument('--network', type=str, default='ProtoNet',
                        help='choose which embedding network to use.'
                             ' ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNet',
                        help='choose which embedding network to use.'
                             ' ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        help='choose which classification head to use.'
                             ' miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--task-embedding', type=str, default='None',
                        help='choose which type of task embedding will be used')
    parser.add_argument('--post-processing', type=str, default='None',
                        help='use an extra post processing net for sample '
                             'embeddings')
    parser.add_argument('--no-film-activation', action='store_true',
                        help='no activation function in FiLM layers')
    parser.add_argument('--dual-BN', action='store_true',
                        help='Use dual BN together with FiLM layers')
    parser.add_argument('--wgrad-prune-ratio', type=float, default=0.0,
                        help='Pruning ratio of the gradient of w')
    parser.add_argument('--film-normalize', action='store_true',
                        help='Normalize the output of FiLM layers')
    parser.add_argument('--no-final-relu', action='store_true',
                        help='No final ReLU layer in the backbone')
    parser.add_argument('--load-naive-backbone', action='store_true',
                        help='Load pre-trained naive backbones')

    opt = parser.parse_args()
    (dataset_test, data_loader) = get_dataset(opt)

    dloader_test = data_loader(
        dataset=dataset_test,
        nKnovel=opt.way,
        nKbase=0,
        nExemplars=opt.shot, # num training examples per novel category
        nTestNovel=opt.query * opt.way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=1,
        epoch_size=opt.episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)

    log_file_path = os.path.join(os.path.dirname(opt.load), "test_log.txt")
    log(log_file_path, str(vars(opt)))

    if 'imagenet' in opt.dataset.lower() and 'film' in opt.task_embedding.lower() \
            and 'rfs' not in opt.network.lower():
        opt.film_indim = 1600
    elif 'rfs' in opt.network.lower():
        opt.film_indim = 640
    else:
        if 'onw' in opt.task_embedding.lower():
            opt.film_indim = 3125
        else:
            opt.film_indim = 2560
    # Define the models
    (embedding_net, cls_head) = get_model(opt)
    add_te_func = get_task_embedding_func(opt)
    postprocessing_net = get_postprocessing_model(opt)
    opt.film_preprocess = 'imagenet' in opt.dataset.lower() and \
                          'film' in opt.task_embedding.lower() and 'rfs' not in opt.network.lower()
    if opt.film_preprocess:
        film_preprocess = nn.Linear(opt.film_preprocess_input_dim,
                                    opt.film_indim, False).cuda()
        device_ids = list(range(len(opt.gpu.split(','))))
        film_preprocess = torch.nn.DataParallel(film_preprocess, device_ids=device_ids)

    # Load saved model checkpoints
    saved_models = torch.load(opt.load)
    if opt.load_naive_backbone and opt.dual_BN:
        from utils import load_from_naive_backbone
        tgt_network = opt.network
        opt.network = tgt_network.split('_')[0]
        src_net, _ = get_model(opt)
        try:
            src_net.load_state_dict(saved_models['embedding'])
        except RuntimeError:
            src_net.module.load_state_dict(saved_models['embedding'])
        load_from_naive_backbone(embedding_net, src_net)
        opt.network = tgt_network
        del src_net
    else:
        embedding_net.load_state_dict(saved_models['embedding'])

    embedding_net.eval()
    cls_head.load_state_dict(saved_models['head'])
    cls_head.eval()
    if 'task_embedding' in saved_models.keys():
        add_te_func.load_state_dict(saved_models['task_embedding'])
        add_te_func.eval()
    if 'postprocessing' in saved_models.keys():
        postprocessing_net.load_state_dict(saved_models['postprocessing'])
        postprocessing_net.eval()
    if 'film_preprocess' in saved_models.keys():
        film_preprocess.load_state_dict(saved_models['film_preprocess'])

    # Evaluate on test set
    test_accuracies = []
    for i, batch in enumerate(tqdm(dloader_test()), 1):
        data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
        # print(labels_support)

        n_support = opt.way * opt.shot
        n_query = opt.way * opt.query

        emb_support = embedding_net(
            data_support.reshape([-1] + list(data_support.shape[-3:])),
            task_embedding=None,
            n_expand=None
        )
        emb_support = emb_support.reshape(1, n_support, -1)

        if i > 1:
            last_emb_task = emb_task
        assert('FiLM' in opt.task_embedding)
        emb_task, _ = add_te_func(
            emb_support, labels_support, opt.way, opt.shot, opt.wgrad_prune_ratio)
        if opt.film_preprocess:
            emb_task = film_preprocess(emb_task.squeeze(1)).unsqueeze(1)

        # Forward pass for support samples with task embeddings
        if emb_task is not None:
            emb_support = embedding_net(
                data_support.reshape([-1] + list(data_support.shape[-3:])),
                task_embedding = emb_task,
                n_expand = n_support
            )
        else:
            emb_support = embedding_net(
                data_support.reshape([-1] + list(data_support.shape[-3:])),
                task_embedding = None,
                n_expand = None
            )
        emb_support = emb_support.reshape(1, n_support, -1)

        # Forward pass for query samples with task embeddings
        if emb_task is not None:
            emb_query = embedding_net(
                data_query.reshape([-1] + list(data_query.shape[-3:])),
                task_embedding = emb_task,
                n_expand = n_query
            )
        else:
            emb_query = embedding_net(
                data_query.reshape([-1] + list(data_query.shape[-3:])),
                task_embedding = None,
                n_expand = None
            )
        emb_query = emb_query.reshape(1, n_query, -1)

        if opt.head == 'SVM':
            logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot, maxIter=3)
            acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1)).item()
        elif opt.head == 'LR':
            assert emb_support.size(0) == 1
            emb_support = normalize(emb_support.squeeze(0)).detach().cpu().numpy()
            emb_query = normalize(emb_query.squeeze(0)).detach().cpu().numpy()
            labels_support = labels_support.view(-1).detach().cpu().numpy()
            labels_query = labels_query.view(-1).detach().cpu().numpy()
            clf = LogisticRegression(random_state=0,
                                     solver='lbfgs',
                                     max_iter=1000,
                                     multi_class='multinomial')
            clf.fit(emb_support, labels_support)
            pred_query = clf.predict(emb_query)
            acc = metrics.accuracy_score(labels_query, pred_query) * 100
        else:
            logits = cls_head(emb_query, emb_support, labels_support, opt.way, opt.shot)
            acc = count_accuracy(logits.reshape(-1, opt.way), labels_query.reshape(-1)).item()

        test_accuracies.append(acc)

        avg = np.mean(np.array(test_accuracies))
        std = np.std(np.array(test_accuracies))
        ci95 = 1.96 * std / np.sqrt(i + 1)

        if i % 50 == 0:
            log(log_file_path, 'Episode [{}/{}]:\t\t\tAccuracy: {:.2f} ± {:.2f} % ({:.2f} %)'\
                  .format(i, opt.episode, avg, ci95, acc))
