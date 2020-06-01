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


def load_from_naive_backbone(tgt_model, src_model):
    # Conv
    tgt_convs = [m for m in tgt_model.modules() if isinstance(m, nn.Conv2d)]
    src_convs = [m for m in src_model.modules() if isinstance(m, nn.Conv2d)]
    assert len(tgt_convs) == len(src_convs)
    for tgt, src in zip(tgt_convs, src_convs):
        assert tgt.weight.size() == src.weight.size()
        tgt.weight.data.copy_(src.weight.data.clone())
        if tgt.bias is not None and src.bias is not None:
            tgt.bias.data.copy_(src.bias.data.clone())
    # DualBN - DualBN.BN_none should copy from naive backbone
    from models.dual_bn import DualBN2d
    tgt_bns = [m for m in tgt_model.modules() if isinstance(m, DualBN2d)]
    src_bns = [m for m in src_model.modules() if isinstance(m, nn.BatchNorm2d)]
    for tgt, src in zip(tgt_bns, src_bns):
        tgt.BN_none.weight.data.copy_(src.weight.data.clone())
        tgt.BN_none.bias.data.copy_(src.bias.data.clone())
        tgt.BN_none.running_mean.data.copy_(src.running_mean.data.clone())
        tgt.BN_none.running_var.data.copy_(src.running_var.data.clone())
        tgt.BN_none.num_batches_tracked.data.copy_(src.num_batches_tracked.data.clone())
        # tgt.BN_none.load_state_dict(src.state_dict())
        # tgt.BN_task.load_state_dict(src.state_dict())
