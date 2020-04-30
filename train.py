# -*- coding: utf-8 -*-
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.classification_heads import ClassificationHead
from models.R2D2_embedding import R2D2Embedding
from models.protonet_embedding import ProtoNetEmbedding
from models.ResNet12_embedding import resnet12
from models.ResNet18_embedding import resnet18
from models.task_embedding import TaskEmbedding
from models.postprocessing import Identity, PostProcessingNet, PostProcessingNetConv1d, PostProcessingNetConv1d_SelfAttn

from utils import set_gpu, Timer, count_accuracy, check_dir, log


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1,index,1)

    return encoded_indicies


def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
            network = resnet12(avg_pool=False,
                               drop_rate=0.1,
                               dropblock_size=5).cuda()
            # device_ids = list(range(len(options.gpu.split(','))))
            # network = torch.nn.DataParallel(network, device_ids=device_ids)
            # network = torch.nn.DataParallel(network, device_ids=[0, 1, 2, 3])
        else:
            network = resnet12(avg_pool=False,
                               drop_rate=0.1,
                               dropblock_size=2).cuda()
        device_ids = list(range(len(options.gpu.split(','))))
        network = torch.nn.DataParallel(network, device_ids=device_ids)
    elif options.network == 'ResNet18':
        assert 'imagenet' in options.dataset.lower()
        network = resnet18(pretrained=False,
                           drop_rate=0.1,
                           dropblock_size=5).cuda()
        device_ids = list(range(len(options.gpu.split(','))))
        network = torch.nn.DataParallel(network, device_ids=device_ids)
    else:
        print("Cannot recognize the network type")
        assert False

    # Choose the classification head
    if options.head == 'ProtoNet':
        cls_head = ClassificationHead(base_learner='ProtoNet').cuda()
    elif options.head == 'Ridge':
        cls_head = ClassificationHead(base_learner='Ridge').cuda()
    elif options.head == 'R2D2':
        cls_head = ClassificationHead(base_learner='R2D2').cuda()
    elif options.head == 'SVM':
        cls_head = ClassificationHead(base_learner='SVM-CS').cuda()
    elif options.head == 'SVM-BiP':
        cls_head = ClassificationHead(base_learner='SVM-CS-BiP').cuda()
    else:
        print ("Cannot recognize the dataset type")
        assert(False)

    return (network, cls_head)


def get_dataset(options):
    # Choose the embedding network
    if options.dataset == 'miniImageNet':
        from data.mini_imagenet import MiniImageNet, FewShotDataloader
        dataset_train = MiniImageNet(phase='train')
        dataset_val = MiniImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'tieredImageNet':
        from data.tiered_imagenet import tieredImageNet, FewShotDataloader
        dataset_train = tieredImageNet(phase='train')
        dataset_val = tieredImageNet(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'CIFAR_FS':
        from data.CIFAR_FS import CIFAR_FS, FewShotDataloader
        dataset_train = CIFAR_FS(phase='train')
        dataset_val = CIFAR_FS(phase='val')
        data_loader = FewShotDataloader
    elif options.dataset == 'FC100':
        from data.FC100 import FC100, FewShotDataloader
        dataset_train = FC100(phase='train')
        dataset_val = FC100(phase='val')
        data_loader = FewShotDataloader
    else:
        print ("Cannot recognize the dataset type")
        assert(False)

    return (dataset_train, dataset_val, data_loader)


def get_task_embedding_func(options):
    # Choose the task embedding function
    te_args = dict(dataset=options.dataset) if options.task_embedding == 'Relation' else dict()
    te_func = TaskEmbedding(metric=options.task_embedding, **te_args).cuda()
    # if options.task_embedding == 'KME':
    #     te_func = TaskEmbedding(metric='KME').cuda()
    # elif options.task_embedding == 'Cosine':
    #     te_func = TaskEmbedding(metric='Cosine').cuda()
    # elif options.task_embedding == 'Entropy_SVM_NoGrad':
    #     te_func = TaskEmbedding(metric='Entropy_SVM_NoGrad').cuda()
    # elif options.task_embedding == 'Entropy_SVM':
    #     te_func = TaskEmbedding(metric='Entropy_SVM').cuda()
    # elif options.task_embedding == 'Cat_SVM_WGrad':
    #     te_func = TaskEmbedding(metric='Cat_SVM_WGrad').cuda()
    # elif options.task_embedding == 'Entropy_Ridge':
    #     te_func = TaskEmbedding(metric='Entropy_Ridge').cuda()
    # elif options.task_embedding == 'Relation':
    #     te_func = TaskEmbedding(metric='Relation', dataset=options.dataset).cuda()
    # elif options.task_embedding == 'None':
    #     te_func = TaskEmbedding(metric='None').cuda()
    # else:
    #     raise ValueError('Cannot recognize the task embedding type `{}`'.format(options.task_embedding))

    device_ids = list(range(len(options.gpu.split(','))))
    te_func = torch.nn.DataParallel(te_func, device_ids=device_ids)

    return te_func


def get_postprocessing_model(options):
    # Choose the post processing network for embeddings
    if options.post_processing == 'FC':
        return PostProcessingNet(dataset=options.dataset, task_embedding=options.task_embedding).cuda()
    if options.post_processing == 'Conv1d':
        postprocessing_net = PostProcessingNetConv1d().cuda()
        # if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
        #     device_ids = list(range(len(options.gpu.split(','))))
        #     postprocessing_net = torch.nn.DataParallel(postprocessing_net, device_ids=device_ids)
        device_ids = list(range(len(options.gpu.split(','))))
        postprocessing_net = torch.nn.DataParallel(postprocessing_net, device_ids=device_ids)
        return postprocessing_net
    if options.post_processing == 'Conv1d_SelfAttn':
        # if options.dataset == 'miniImageNet' or options.dataset == 'tieredImageNet':
        #     skip_attn1 = True
        #     print('First attention skipped')
        # else:
        #     skip_attn1 = False
        postprocessing_net = PostProcessingNetConv1d_SelfAttn(dataset=options.dataset).cuda()
        device_ids = list(range(len(options.gpu.split(','))))
        postprocessing_net = torch.nn.DataParallel(postprocessing_net, device_ids=device_ids)
        return postprocessing_net
    elif options.post_processing == 'None':
        return Identity().cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epoch', type=int, default=60,
                            help='number of training epochs')
    parser.add_argument('--save-epoch', type=int, default=10,
                            help='frequency of model saving')
    parser.add_argument('--train-shot', type=int, default=15,
                            help='number of support examples per training class')
    parser.add_argument('--val-shot', type=int, default=5,
                            help='number of support examples per validation class')
    parser.add_argument('--train-query', type=int, default=6,
                            help='number of query examples per training class')
    parser.add_argument('--val-episode', type=int, default=2000,
                            help='number of episodes per validation')
    parser.add_argument('--val-query', type=int, default=15,
                            help='number of query examples per validation class')
    parser.add_argument('--train-way', type=int, default=5,
                            help='number of classes in one training episode')
    parser.add_argument('--test-way', type=int, default=5,
                            help='number of classes in one test (or validation) episode')
    parser.add_argument('--save-path', default='./experiments/exp_1')
    parser.add_argument('--load', default=None, help='path of the checkpoint file')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--network', type=str, default='ProtoNet',
                            help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNet',
                            help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                            help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                            help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                            help='epsilon of label smoothing')
    parser.add_argument('--task-embedding', type=str, default='None',
                            help='choose which type of task embedding will be used')
    parser.add_argument('--start-epoch', type=int, default=-1,
                            help='choose when to use task embedding')
    parser.add_argument('--post-processing', type=str, default='None',
                            help='use an extra post processing net for sample embeddings')

    opt = parser.parse_args()

    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot, # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=4,
        epoch_size=opt.episodes_per_batch * 1000, # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot, # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way, # num test examples for all the novel categories
        nTestBase=0, # num test examples for all the base categories
        batch_size=1,
        num_workers=0,
        epoch_size=1 * opt.val_episode, # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    add_te_func = get_task_embedding_func(opt)
    postprocessing_net = get_postprocessing_model(opt)

    optimizer = torch.optim.SGD([{'params': embedding_net.parameters()},
                                 {'params': cls_head.parameters()},
                                 {'params': add_te_func.parameters()},
                                 {'params': postprocessing_net.parameters()}],
                                lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Load saved model checkpoints
    if opt.load is not None:
        saved_models = torch.load(opt.load)
        # NOTE: there is a `-1` because `epoch` starts counting from 1
        last_epoch = saved_models['epoch'] - 1 if 'epoch' in saved_models.keys() else -1
        embedding_net.load_state_dict(saved_models['embedding'])
        cls_head.load_state_dict(saved_models['head'])
        if 'task_embedding' in saved_models.keys():
            add_te_func.load_state_dict(saved_models['task_embedding'])
        if 'postprocessing' in saved_models.keys():
            postprocessing_net.load_state_dict(saved_models['postprocessing'])
        if 'optimizer' in saved_models.keys():
            optimizer.load_state_dict(saved_models['optimizer'])
    else:
        last_epoch = -1

    lambda_epoch = lambda e: 1.0 if e < 20 else (0.06 if e < 40 else 0.012 if e < 50 else (0.0024))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda_epoch,
                                                     last_epoch=last_epoch)

    max_val_acc = 0.0

    timer = Timer()
    x_entropy = torch.nn.CrossEntropyLoss()

    for epoch in range(last_epoch + 2, opt.num_epoch + 1):
        # Train on the training split
        # Learning rate decay
        lr_scheduler.step()

        # Fetch the current epoch's learning rate
        epoch_learning_rate = 0.1
        for param_group in optimizer.param_groups:
            epoch_learning_rate = param_group['lr']

        log(log_file_path, 'Train Epoch: {}\tLearning Rate: {:.4f}'.format(
                            epoch, epoch_learning_rate))

        # _, _ = [x.train() for x in (embedding_net, cls_head)]
        _, _, _, _ = [x.train() for x in (embedding_net, cls_head, add_te_func, postprocessing_net)]

        train_accuracies = []
        train_losses = []

        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]
            # print(data_support.size())
            # print(labels_support)

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)

            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)

            # print(emb_support.size(), emb_query.size())

            if epoch >= opt.start_epoch:
                emb_support, emb_query, G_support, G_query = add_te_func(
                    emb_support, emb_query, data_support, data_query,
                    labels_support, opt.train_way, opt.train_shot
                )

            emb_support = postprocessing_net(emb_support.reshape([-1] + list(emb_support.size()[2:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch, train_n_support, -1)
            emb_query = postprocessing_net(emb_query.reshape([-1] + list(emb_query.size()[2:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query, -1)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.train_way, opt.train_shot)

            smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
            smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

            log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way), dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()

            acc = count_accuracy(logit_query.reshape(-1, opt.train_way), labels_query.reshape(-1))

            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if (i % 100 == 0):
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path, 'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                            epoch, i, len(dloader_train), loss.item(), train_acc_avg, acc))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on the validation split
        _, _, _, _ = [x.eval() for x in (embedding_net, cls_head, add_te_func, postprocessing_net)]

        val_accuracies = []
        val_losses = []

        for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [x.cuda() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query

            emb_support = embedding_net(data_support.reshape([-1] + list(data_support.shape[-3:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            if epoch >= opt.start_epoch:
                emb_support, emb_query, _, _ = add_te_func(
                    emb_support, emb_query, data_support, data_query,
                    labels_support, opt.test_way, opt.val_shot
                )
                # emb_query, _ = add_te_func(emb_query, emb_support)
                # emb_support, _ = add_te_func(emb_support, emb_support)

            emb_support = postprocessing_net(emb_support.reshape([-1] + list(emb_support.size()[2:])))
            emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_query = postprocessing_net(emb_query.reshape([-1] + list(emb_query.size()[2:])))
            emb_query = emb_query.reshape(1, test_n_query, -1)

            # emb_support = postprocessing_net(emb_support)
            # emb_query = postprocessing_net(emb_query)

            logit_query = cls_head(emb_query, emb_support, labels_support, opt.test_way, opt.val_shot)

            loss = x_entropy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))

            val_accuracies.append(acc.item())
            val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save({'epoch': epoch,
                        'embedding': embedding_net.state_dict(),
                        'head': cls_head.state_dict(),
                        'task_embedding': add_te_func.state_dict(),
                        'postprocessing': postprocessing_net.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(opt.save_path, 'best_model.pth'))
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path, 'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %'\
                  .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        torch.save({'epoch': epoch,
                    'embedding': embedding_net.state_dict(),
                    'head': cls_head.state_dict(),
                    'task_embedding': add_te_func.state_dict(),
                    'postprocessing': postprocessing_net.state_dict(),
                    'optimizer': optimizer.state_dict()},
                   os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0 or epoch in [20,21,22,23,24,25]:
            torch.save({'epoch': epoch,
                        'embedding': embedding_net.state_dict(),
                        'head': cls_head.state_dict(),
                        'task_embedding': add_te_func.state_dict(),
                        'postprocessing': postprocessing_net.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(timer.measure(), timer.measure(epoch / float(opt.num_epoch))))
