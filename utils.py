import os
import time
import pprint
import torch
import torch.nn as nn


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def check_dir(path):
    """
    Create directory if it does not exist.
        path:           Path of directory.
    """
    if not os.path.exists(path):
        os.mkdir(path)


def count_accuracies(logits, labels):
    # `logits` of shape (episodes_per_batch, n_query, n_ways)
    # `labels` of shape (episodes_per_batch, n_query)
    pred = torch.argmax(logits, dim=2)
    # labels = labels.view(-1)
    accuracies = 100 * pred.eq(labels).float().mean(dim=1)
    return accuracies


def count_accuracy(logits, labels):
    pred = torch.argmax(logits, dim=1).view(-1)
    labels = labels.view(-1)
    accuracy = 100 * pred.eq(labels).float().mean()
    return accuracy


class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / float(p)
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def log(log_file_path, string):
    """
    Write one line of log into screen and file.
        log_file_path: Path of log file.
        string:        String to write in log file.
    """
    with open(log_file_path, 'a+') as f:
        f.write(string + '\n')
        f.flush()
    print(string)


def load_dual_bn_from_naive_backbone(tgt_model, src_model):
    from models.dual_bn import DualBN2d
    tgt_bns = [m for m in tgt_model.modules() if isinstance(m, DualBN2d)]
    src_bns = [m for m in src_model.modules() if isinstance(m, nn.BatchNorm2d)]
    for tgt, src in zip(tgt_bns, src_bns):
        tgt.BN_none.load_state_dict(src.state_dict())
        # tgt.BN_task.load_state_dict(src.state_dict())
