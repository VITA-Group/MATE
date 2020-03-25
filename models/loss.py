import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .FiLM import FiLM_Layer


def get_film_loss(model, loss_type):
    if 'none' in loss_type.lower():
        return 0.0
    elif 'ortho' in loss_type.lower():
        return get_film_srip_loss(model)
    elif 'msgan' in loss_type.lower():
        return get_film_msgan_loss(model)
    elif 'dsgan' in loss_type.lower():
        return get_film_dsgan_loss(model)
    else:
        raise ValueError('Invalid `loss_type` for FiLM regularization')


def get_film_msgan_loss(model):
    """Short summary.

    Parameters
    ----------
    model : type
        Description of parameter `model`.

    Returns
    -------
    type
        Description of returned object.

    """

    film_loss = 0.0
    for m in model.modules():
        if isinstance(m, FiLM_Layer):
            assert m._task_embs is not None

            tasks_per_batch = m._task_emb.size(0) # (tasks_per_batch, 1, d)

            _task_emb = m._task_emb.expand(-1, tasks_per_batch, -1)
            _film_out = m._film_out.expand(-1, tasks_per_batch, -1)

            d_input  = torch.dist(_task_emb, _task_emb.transpose(0,1), p = 1)
            d_output = torch.dist(_film_out, _film_out.transpose(0,1), p = 1)
            lz = d_output / d_input
            eps = 1 * 1e-5
            film_loss += 1 / (lz + eps)

    return film_loss
