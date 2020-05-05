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
from models.ResNet12_FiLM_embedding import resnet12_film
from models.task_embedding import TaskEmbedding
from models.postprocessing import Identity, PostProcessingNet, \
    PostProcessingNetConv1d, PostProcessingNetConv1d_SelfAttn
from models.loss import get_film_loss

from utils import set_gpu, Timer, count_accuracy, count_accuracies, check_dir, \
    log


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
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def get_model(options):
    # Choose the embedding network
    if options.network == 'ProtoNet':
        network = ProtoNetEmbedding().cuda()
    elif options.network == 'R2D2':
        network = R2D2Embedding().cuda()
    elif options.network == 'ResNet':
        if 'imagenet' in opt.dataset.lower():
            network = resnet12(avg_pool=False,
                               drop_rate=0.1,
                               dropblock_size=5).cuda()
        else:
            network = resnet12(avg_pool=False,
                               drop_rate=0.1,
                               dropblock_size=2).cuda()
        device_ids = list(range(len(options.gpu.split(','))))
        network = torch.nn.DataParallel(network, device_ids=device_ids)
    elif options.network == 'ResNet_FiLM':
        film_act = None if options.no_film_activation else F.leaky_relu
        if 'imagenet' in opt.dataset.lower():
            network = resnet12_film(
                avg_pool=False, drop_rate=0.1, dropblock_size=5,
                film_indim=2560, film_alpha=1.0, film_act=film_act,
                final_relu=(not options.no_final_relu),
                film_normalize=options.film_normalize,
                dual_BN=options.dual_BN).cuda()
            options.film_preprocess_input_dim = 16000
        else:
            network = resnet12_film(
                avg_pool=False, drop_rate=0.1, dropblock_size=2,
                film_indim=2560, film_alpha=1.0, film_act=film_act,
                final_relu=(not options.no_final_relu),
                film_normalize=options.film_normalize,
                dual_BN=options.dual_BN).cuda()
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
        print("Cannot recognize the dataset type")
        assert (False)

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
        print("Cannot recognize the dataset type")
        assert (False)

    return (dataset_train, dataset_val, data_loader)


def get_task_embedding_func(options):
    # Choose the task embedding function
    te_args = dict(
        dataset=options.dataset) if options.task_embedding == 'Relation' else dict()
    te_func = TaskEmbedding(metric=options.task_embedding, **te_args).cuda()

    device_ids = list(range(len(options.gpu.split(','))))
    te_func = torch.nn.DataParallel(te_func, device_ids=device_ids)

    return te_func


def get_postprocessing_model(options):
    # Choose the post processing network for embeddings
    if options.post_processing == 'FC':
        return PostProcessingNet(dataset=options.dataset,
                                 task_embedding=options.task_embedding).cuda()
    if options.post_processing == 'Conv1d':
        postprocessing_net = PostProcessingNetConv1d().cuda()
        device_ids = list(range(len(options.gpu.split(','))))
        postprocessing_net = torch.nn.DataParallel(postprocessing_net,
                                                   device_ids=device_ids)
        return postprocessing_net
    if options.post_processing == 'Conv1d_SelfAttn':
        postprocessing_net = PostProcessingNetConv1d_SelfAttn(
            dataset=options.dataset).cuda()
        device_ids = list(range(len(options.gpu.split(','))))
        postprocessing_net = torch.nn.DataParallel(postprocessing_net,
                                                   device_ids=device_ids)
        return postprocessing_net
    elif options.post_processing == 'None':
        return Identity().cuda()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
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
    parser.add_argument('--load', default=None,
                        help='path of the checkpoint file')
    parser.add_argument('--gpu', default='0, 1, 2, 3')
    parser.add_argument('--network', type=str, default='ProtoNet',
                        help='choose which embedding network to use. ProtoNet, R2D2, ResNet')
    parser.add_argument('--head', type=str, default='ProtoNet',
                        help='choose which classification head to use. ProtoNet, Ridge, R2D2, SVM')
    parser.add_argument('--dataset', type=str, default='miniImageNet',
                        help='choose which classification head to use. miniImageNet, tieredImageNet, CIFAR_FS, FC100')
    parser.add_argument('--episodes-per-batch', type=int, default=8,
                        help='number of episodes per batch')
    parser.add_argument('--val-episodes-per-batch', type=int, default=20,
                        help='number of episodes per batch')
    parser.add_argument('--eps', type=float, default=0.0,
                        help='epsilon of label smoothing')
    parser.add_argument('--task-embedding', type=str, default='None',
                        help='choose which type of task embedding will be used')
    parser.add_argument('--start-epoch', type=int, default=-1,
                        help='choose when to use task embedding')
    parser.add_argument('--post-processing', type=str, default='None',
                        help='use an extra post processing net for sample embeddings')
    parser.add_argument('--no-film-activation', action='store_true',
                        help='no activation function in FiLM layers')
    parser.add_argument('--dual-BN', action='store_true',
                        help='Use dual BN together with FiLM layers')
    parser.add_argument('--mix-train', action='store_true',
                        help='Mix train using logits without and with task embedding')
    parser.add_argument('--orthogonal-reg', type=float, default=0.0,
                        help='Regularization term of orthogonality between task representations')
    parser.add_argument('--wgrad-l1-reg', type=float, default=0.0,
                        help='Regularization term of l1 norm of WGrad')
    parser.add_argument('--wgrad-prune-ratio', type=float, default=0.0,
                        help='Pruning ratio of the gradient of w')
    parser.add_argument('--film-reg-type', type=str, default='None',
                        help='Regularization on FiLM layers')
    parser.add_argument('--film-reg-level', type=float, default=0.0,
                        help='Coefficient of the regularization term')
    parser.add_argument('--fix-film', action='store_true',
                        help='Fix FiLM layers in training')
    parser.add_argument('--train-film-dualBN', action='store_true',
                        help='Train FiLM layers and DualBN only during training')
    parser.add_argument('--film-normalize', action='store_true',
                        help='Normalize the output of FiLM layers')
    parser.add_argument('--no-final-relu', action='store_true',
                        help='No final ReLU layer in the backbone')
    parser.add_argument('--load-naive-backbone', action='store_true',
                        help='Load pre-trained naive backbones')

    opt = parser.parse_args()

    (dataset_train, dataset_val, data_loader) = get_dataset(opt)

    # Dataloader of Gidaris & Komodakis (CVPR 2018)
    dloader_train = data_loader(
        dataset=dataset_train,
        nKnovel=opt.train_way,
        nKbase=0,
        nExemplars=opt.train_shot,  # num training examples per novel category
        nTestNovel=opt.train_way * opt.train_query,
        # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        batch_size=opt.episodes_per_batch,
        num_workers=8,
        epoch_size=opt.episodes_per_batch * 1000,  # num of batches per epoch
    )

    dloader_val = data_loader(
        dataset=dataset_val,
        nKnovel=opt.test_way,
        nKbase=0,
        nExemplars=opt.val_shot,  # num training examples per novel category
        nTestNovel=opt.val_query * opt.test_way,
        # num test examples for all the novel categories
        nTestBase=0,  # num test examples for all the base categories
        # batch_size=1,
        batch_size=opt.val_episodes_per_batch,
        num_workers=0,
        epoch_size=1 * opt.val_episode,  # num of batches per epoch
    )

    set_gpu(opt.gpu)
    check_dir('./experiments/')
    check_dir(opt.save_path)

    log_file_path = os.path.join(opt.save_path, "train_log.txt")
    log(log_file_path, str(vars(opt)))

    (embedding_net, cls_head) = get_model(opt)
    add_te_func = get_task_embedding_func(opt)
    postprocessing_net = get_postprocessing_model(opt)
    if 'imagenet' in opt.dataset.lower() and 'film' in opt.task_embedding.lower():
        film_preprocess = nn.Linear(opt.film_preprocess_input_dim, 2560, False).cuda()

    if opt.train_film_dualBN:
        assert not opt.fix_film
        from models.dual_bn import DualBN2d
        from models.FiLM import FiLM_Layer
        embedding_params = []
        for m in embedding_net.modules():
            if isinstance(m, DualBN2d):
                embedding_params += list(m.BN_task.parameters())
            if isinstance(m, FiLM_Layer):
                embedding_params += list(m.parameters())
    else:
        embedding_params = embedding_net.parameters()

    params = [
        {'params': embedding_params},
        {'params': cls_head.parameters()},
        {'params': add_te_func.parameters()},
        {'params': postprocessing_net.parameters()}
    ]

    if ('imagenet' in opt.dataset.lower() and
            'film' in opt.task_embedding.lower()):
        params.append({'params': film_preprocess.parameters()})

    optimizer = torch.optim.SGD(
        params, lr=opt.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    # Load saved model checkpoints
    if opt.load is not None:
        saved_models = torch.load(opt.load)
        # NOTE: there is a `-1` because `epoch` starts counting from 1
        last_epoch = saved_models[
                         'epoch'] - 1 if 'epoch' in saved_models.keys() else -1
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
            # src_net = None
        else:
            embedding_net.load_state_dict(saved_models['embedding'])
        cls_head.load_state_dict(saved_models['head'])
        if 'task_embedding' in saved_models.keys():
            add_te_func.load_state_dict(saved_models['task_embedding'])
        if 'postprocessing' in saved_models.keys():
            postprocessing_net.load_state_dict(saved_models['postprocessing'])
        if 'optimizer' in saved_models.keys():
            optimizer.load_state_dict(saved_models['optimizer'])
        if 'film_preprocess' in saved_models.keys():
            film_preprocess.load_state_dict(saved_models['film_preprocess'])
    else:
        last_epoch = -1

    if opt.fix_film:
        new_param_list = [
            param for param in optimizer.param_groups[0]['params']
            if len(param.size()) != 2
        ]
        optimizer.param_groups[0]['params'] = new_param_list

    lambda_epoch = lambda e: 1.0 if e < 20 else (
        0.06 if e < 40 else 0.012 if e < 50 else 0.0024)
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
        _, _, _, _ = [x.train() for x in (embedding_net, cls_head,
                                          add_te_func, postprocessing_net)]
        if opt.train_film_dualBN:
            for m in embedding_net.modules():
                if isinstance(m, DualBN2d):
                    m.BN_none.eval()
        if 'imagenet' in opt.dataset.lower():
            film_preprocess.train()

        train_accuracies = []
        train_losses = []

        for i, batch in enumerate(tqdm(dloader_train(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [
                x.cuda() for x in batch]
            # print(data_support.size())

            train_n_support = opt.train_way * opt.train_shot
            train_n_query = opt.train_way * opt.train_query

            # First pass without task embeddings
            emb_support = embedding_net(
                data_support.reshape([-1] + list(data_support.shape[-3:])),
                task_embedding=None,
                n_expand=None
            )
            emb_support = emb_support.reshape(
                opt.episodes_per_batch, train_n_support, -1)

            if opt.mix_train:
                emb_query_none = embedding_net(
                    data_query.reshape([-1] + list(data_query.shape[-3:])),
                    task_embedding=None,
                    n_expand=None
                )
                emb_query_none = emb_query_none.reshape(opt.episodes_per_batch,
                                                        train_n_query, -1)
                logit_query_none = cls_head(emb_query_none, emb_support,
                                            labels_support, opt.train_way,
                                            opt.train_shot)

            if epoch > opt.start_epoch:
                assert ('FiLM' in opt.task_embedding)
                emb_task, _ = add_te_func(
                    emb_support, labels_support, opt.train_way, opt.train_shot,
                    opt.wgrad_prune_ratio)
                if 'imagenet' in opt.dataset.lower():
                    emb_task = film_preprocess(emb_task.squeeze(1)).unsqueeze(1)
            else:
                emb_task, _ = None, None

            if opt.orthogonal_reg > 0 and emb_task is not None:
                mask = 1.0 - torch.eye(emb_task.size(0)).float().cuda()
                emb_task_temp = emb_task.squeeze(1)
                gram = torch.matmul(emb_task_temp, emb_task_temp.t())
                loss_ortho_reg = ((gram * mask) ** 2.0).sum()
            else:
                mask = 0.0
                emb_task_temp = 0.0
                gram = 0.0
                loss_ortho_reg = 0.0

            # if opt.wgrad_l1_reg > 0 and G is not None:
            #     loss_wgrad_l1_reg = G.sum()
            # else:
            #     loss_wgrad_l1_reg = 0.0

            # Forward pass for support samples with task embeddings
            if emb_task is not None:
                # emb_task_support_batch = emb_task.expand(-1, train_n_support, -1)
                emb_support = embedding_net(
                    data_support.reshape([-1] + list(data_support.shape[-3:])),
                    # emb_task_support_batch.reshape(-1, emb_task.size(-1)),
                    task_embedding=emb_task,
                    n_expand=train_n_support
                )
            else:
                emb_support = embedding_net(
                    data_support.reshape([-1] + list(data_support.shape[-3:])),
                    task_embedding=None,
                    n_expand=None
                )
            # emb_support = postprocessing_net(emb_support.reshape([-1] + list(emb_support.size()[2:])))
            emb_support = emb_support.reshape(opt.episodes_per_batch,
                                              train_n_support, -1)

            # Forward pass for query samples with task embeddings
            if emb_task is not None:
                # emb_task_query_batch = emb_task.expand(-1, train_n_query, -1)
                emb_query = embedding_net(
                    data_query.reshape([-1] + list(data_query.shape[-3:])),
                    task_embedding=emb_task,
                    n_expand=train_n_query
                )
            else:
                emb_query = embedding_net(
                    data_query.reshape([-1] + list(data_query.shape[-3:])),
                    task_embedding=None,
                    n_expand=None
                )
            # emb_query = postprocessing_net(emb_query.reshape([-1] + list(emb_query.size()[2:])))
            emb_query = emb_query.reshape(opt.episodes_per_batch, train_n_query,
                                          -1)

            logit_query = cls_head(emb_query, emb_support, labels_support,
                                   opt.train_way, opt.train_shot)

            smoothed_one_hot = one_hot(labels_query.reshape(-1), opt.train_way)
            smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (
                        1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)

            log_prb = F.log_softmax(logit_query.reshape(-1, opt.train_way),
                                    dim=1)
            loss = -(smoothed_one_hot * log_prb).sum(dim=1)
            loss = loss.mean()
            if opt.mix_train:
                alpha = 0.99 * 0.5 ** (epoch // 4)
                log_prb_none = F.log_softmax(
                    logit_query_none.reshape(-1, opt.train_way), dim=1)
                loss_none = -(smoothed_one_hot * log_prb_none).sum(dim=1).mean()
                loss = loss_none * alpha + loss * (1 - alpha)

            # FiLM regularization
            loss_film_reg = get_film_loss(
                embedding_net,
                emb_task,
                opt.film_reg_type
            )
            loss += opt.film_reg_level * loss_film_reg

            # print(loss_ortho_reg)
            loss += opt.orthogonal_reg * loss_ortho_reg
            # loss += opt.orthogonal_reg * loss_ortho_reg + opt.wgrad_l1_reg * loss_wgrad_l1_reg

            acc = count_accuracy(logit_query.reshape(-1, opt.train_way),
                                 labels_query.reshape(-1))

            train_accuracies.append(acc.item())
            train_losses.append(loss.item())

            if i % 100 == 0:
                train_acc_avg = np.mean(np.array(train_accuracies))
                log(log_file_path,
                    'Train Epoch: {}\tBatch: [{}/{}]\tLoss: {:.4f}\tAccuracy: {:.2f} % ({:.2f} %)'.format(
                        epoch, i, len(dloader_train), loss.item(),
                        train_acc_avg, acc.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # empty cache
        del data_support, labels_support, data_query, labels_query
        del emb_support, emb_task, emb_task_temp, gram, mask
        del emb_query, logit_query, log_prb, smoothed_one_hot
        del train_losses, train_accuracies, acc, loss, loss_film_reg, loss_ortho_reg
        torch.cuda.empty_cache()

        # Evaluate on the validation split
        _, _, _, _ = [x.eval() for x in
                      (embedding_net, cls_head, add_te_func, postprocessing_net)]
        if 'imagenet' in opt.dataset.lower():
            film_preprocess.eval()

        val_accuracies = []
        val_losses = []

        for i, batch in enumerate(tqdm(dloader_val(epoch)), 1):
            data_support, labels_support, data_query, labels_query, _, _ = [
                x.cuda() for x in batch]

            test_n_support = opt.test_way * opt.val_shot
            test_n_query = opt.test_way * opt.val_query

            emb_support = embedding_net(
                data_support.reshape([-1] + list(data_support.shape[-3:])),
                task_embedding=None,
                n_expand=None
            )
            # emb_support = emb_support.reshape(1, test_n_support, -1)
            emb_support = emb_support.reshape(opt.val_episodes_per_batch,
                                              test_n_support, -1)
            # emb_query = embedding_net(data_query.reshape([-1] + list(data_query.shape[-3:])))
            # emb_query = emb_query.reshape(1, test_n_query, -1)

            if epoch > opt.start_epoch:
                assert ('FiLM' in opt.task_embedding)
                emb_task, G = add_te_func(
                    emb_support, labels_support, opt.test_way, opt.val_shot,
                    opt.wgrad_prune_ratio)
                if 'imagenet' in opt.dataset.lower():
                    emb_task = film_preprocess(emb_task.squeeze(1)).unsqueeze(1)
            else:
                emb_task, G = None, None

            # Forward pass for support samples with task embeddings
            if emb_task is not None:
                # emb_task_support_batch = emb_task.expand(-1, test_n_support, -1)
                emb_support = embedding_net(
                    data_support.reshape([-1] + list(data_support.shape[-3:])),
                    task_embedding=emb_task,
                    n_expand=test_n_support
                )
            else:
                emb_support = embedding_net(
                    data_support.reshape([-1] + list(data_support.shape[-3:])),
                    task_embedding=None,
                    n_expand=None
                )
            # emb_support = postprocessing_net(emb_support.reshape([-1] + list(emb_support.size()[2:])))
            emb_support = emb_support.reshape(opt.val_episodes_per_batch,
                                              test_n_support, -1)

            # Forward pass for query samples with task embeddings
            if emb_task is not None:
                # emb_task_query_batch = emb_task.expand(-1, test_n_query, -1)
                emb_query = embedding_net(
                    data_query.reshape([-1] + list(data_query.shape[-3:])),
                    task_embedding=emb_task,
                    n_expand=test_n_query
                )
            else:
                emb_query = embedding_net(
                    data_query.reshape([-1] + list(data_query.shape[-3:])),
                    task_embedding=None,
                    n_expand=None
                )
            # emb_query = postprocessing_net(emb_query.reshape([-1] + list(emb_query.size()[2:])))
            emb_query = emb_query.reshape(opt.val_episodes_per_batch,
                                          test_n_query, -1)

            # emb_support = postprocessing_net(emb_support)
            # emb_query = postprocessing_net(emb_query)

            logit_query = cls_head(emb_query, emb_support, labels_support,
                                   opt.test_way, opt.val_shot)

            loss = x_entropy(logit_query.reshape(-1, opt.test_way),
                             labels_query.reshape(-1))
            # acc = count_accuracy(logit_query.reshape(-1, opt.test_way), labels_query.reshape(-1))
            accs = count_accuracies(logit_query, labels_query)

            val_accuracies += accs.detach().cpu().tolist()
            # val_accuracies.append(acc.item())
            val_losses.append(loss.item())

        val_acc_avg = np.mean(np.array(val_accuracies))
        val_acc_ci95 = 1.96 * np.std(np.array(val_accuracies)) / np.sqrt(
            opt.val_episode)

        val_loss_avg = np.mean(np.array(val_losses))

        save_dict = {
            'epoch': epoch,
            'embedding': embedding_net.state_dict(),
            'head': cls_head.state_dict(),
            'task_embedding': add_te_func.state_dict(),
            'postprocessing': postprocessing_net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        if 'imagenet' in opt.dataset.lower() and 'film' in opt.task_embedding.lower():
            save_dict['film_preprocess'] = film_preprocess.state_dict()
        if val_acc_avg > max_val_acc:
            max_val_acc = val_acc_avg
            torch.save(save_dict, os.path.join(opt.save_path, 'best_model.pth'))
            # torch.save({'epoch': epoch,
            #             'embedding': embedding_net.state_dict(),
            #             'head': cls_head.state_dict(),
            #             'task_embedding': add_te_func.state_dict(),
            #             'postprocessing': postprocessing_net.state_dict(),
            #             'optimizer': optimizer.state_dict()},
            #            os.path.join(opt.save_path, 'best_model.pth'))
            log(log_file_path,
                'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} % (Best)' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))
        else:
            log(log_file_path,
                'Validation Epoch: {}\t\t\tLoss: {:.4f}\tAccuracy: {:.2f} ± {:.2f} %' \
                .format(epoch, val_loss_avg, val_acc_avg, val_acc_ci95))

        torch.save(save_dict, os.path.join(opt.save_path, 'last_epoch.pth'))
        # torch.save({'epoch': epoch,
        #             'embedding': embedding_net.state_dict(),
        #             'head': cls_head.state_dict(),
        #             'task_embedding': add_te_func.state_dict(),
        #             'postprocessing': postprocessing_net.state_dict(),
        #             'optimizer': optimizer.state_dict()},
        #            os.path.join(opt.save_path, 'last_epoch.pth'))

        if epoch % opt.save_epoch == 0 or epoch in [21, 22, 23, 24, 25]:
            torch.save(save_dict, os.path.join(opt.save_path,
                                               'epoch_{}.pth'.format(epoch)))
            # torch.save({'epoch': epoch,
            #             'embedding': embedding_net.state_dict(),
            #             'head': cls_head.state_dict(),
            #             'task_embedding': add_te_func.state_dict(),
            #             'postprocessing': postprocessing_net.state_dict(),
            #             'optimizer': optimizer.state_dict()},
            #            os.path.join(opt.save_path, 'epoch_{}.pth'.format(epoch)))

        log(log_file_path, 'Elapsed Time: {}/{}\n'.format(
            timer.measure(), timer.measure(epoch / float(opt.num_epoch))))

        # empty cache
        del data_support, labels_support, data_query, labels_query
        del emb_support, emb_task, G, emb_task_temp, gram, mask
        del emb_query, logit_query, log_prb, smoothed_one_hot
        del val_losses, val_acc_avg, val_acc_ci95, val_accuracies, accs, loss
        torch.cuda.empty_cache()
