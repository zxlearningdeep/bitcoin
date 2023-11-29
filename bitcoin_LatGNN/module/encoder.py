# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.preprocess import sparse_mx_to_torch_sparse_tensor, normalize_adj_from_tensor, add_self_loop_and_normalize, remove_self_loop

EPS = 1e-15

def APPNP(h, adj, nlayer, alpha):

    h_0 = h
    z = h
    for i in range(nlayer):
        z = torch.sparse.mm(adj, z)
        z = (1 - alpha) * z + alpha * h_0
    z = F.normalize(z, dim=1, p=2)

    return z
