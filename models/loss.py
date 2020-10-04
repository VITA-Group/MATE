import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .FiLM import FiLM_Layer


def get_film_loss(model, task_emb, loss_type):
    if 'none' in loss_type.lower():
        return 0.0
    elif 'msgan' in loss_type.lower():
        return get_film_msgan_loss(model, task_emb)
    else:
        raise ValueError('Invalid `loss_type` for FiLM regularization')


def get_film_msgan_loss(model, task_emb, detach=True):
    """Short summary.

    Parameters
    ----------
    model : type
        Description of parameter `model`.
    task_emb : type
        Description of parameter `task_emb`.

    Returns
    -------
    type
        Description of returned object.

    """

    assert task_emb is not None
    if detach:
        task_emb = task_emb.detach()
    tasks_per_batch = task_emb.size(0) # (tasks_per_batch, 1, d)
    task_emb_expanded = task_emb.expand(-1, tasks_per_batch, -1)
    film_loss = 0.0

    for m in model.modules():
        if isinstance(m, FiLM_Layer):
            film_out = m.get_mlp_output(task_emb)
            film_out_expanded = film_out.expand(-1, tasks_per_batch, -1)

            d_input  = torch.dist(task_emb_expanded,
                                  task_emb_expanded.transpose(0,1),
                                  p = 1)
            d_output = torch.dist(film_out_expanded,
                                  film_out_expanded.transpose(0,1),
                                  p = 1)
            lz = d_output / d_input
            eps = 1 * 1e-5
            film_loss += 1 / (lz + eps)

    return film_loss
