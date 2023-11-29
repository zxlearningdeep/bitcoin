import numpy as np
import torch
import torch as th
import warnings
import matplotlib.pyplot as plt
import datetime
import pickle as pkl
import os
import random
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from utils.load_data import load_data
from utils.params import set_params
from module.preprocess import remove_self_loop
warnings.filterwarnings('ignore')

EPS = 1e-15


def get_knn(x, k):
    x_norm = torch.norm(x, dim=-1, keepdim=True)
    zero_indices = torch.nonzero(x_norm.flatten() == 0)
    x_norm[zero_indices] += EPS
    dot_numerator = torch.mm(x, x.t())
    dot_denominator = torch.mm(x_norm, x_norm.t())
    sim_matrix = dot_numerator / dot_denominator

    sim_diag = torch.diag_embed(torch.diag(sim_matrix))
    sim_matrix = sim_matrix - sim_diag
    _, k_indices_pos = torch.topk(sim_matrix, k=k, dim=1)
    _, k_indices_neg = torch.topk(-sim_matrix, k=k, dim=1)

    source = torch.tensor(range(len(x))).reshape(-1,1).to(x.device)
    k_source = source.repeat(1,k).flatten()

    k_indices_pos = k_indices_pos.flatten()
    k_indices_pos = torch.stack((k_source,k_indices_pos),dim=0)

    k_indices_neg = k_indices_neg.flatten()
    k_indices_neg = torch.stack((k_source,k_indices_neg),dim=0)

    kg_pos = torch.sparse.FloatTensor(k_indices_pos, torch.ones((len(k_indices_pos[0]))).to(x.device), ([len(x),len(x)]))
    kg_eng = torch.sparse.FloatTensor(k_indices_neg, torch.ones((len(k_indices_neg[0]))).to(x.device), ([len(x),len(x)]))

    return kg_pos, kg_eng


def get_top_k(sim_l, sim_h, k):
    _, k_indices_pos = torch.topk(sim_l, k=k, dim=1)
    _, k_indices_neg = torch.topk(sim_h, k=k, dim=1)

    source = torch.tensor(range(len(x))).reshape(-1, 1).to(x.device)
    k_source = source.repeat(1, k).flatten()

    k_indices_pos = k_indices_pos.flatten()
    k_indices_pos = torch.stack((k_source, k_indices_pos), dim=0)

    k_indices_neg = k_indices_neg.flatten()
    k_indices_neg = torch.stack((k_source, k_indices_neg), dim=0)

    # kg_pos = torch.sparse.FloatTensor(k_indices_pos, torch.ones((len(k_indices_pos[0]))).to(x.device), ([len(x), len(x)]))
    # kg_neg = torch.sparse.FloatTensor(k_indices_neg, torch.ones((len(k_indices_neg[0]))).to(x.device), ([len(x), len(x)]))

    homo_r_l = len(torch.nonzero(label[k_indices_pos[0]] == label[k_indices_pos[1]])) / len(k_indices_pos[0])
    homo_r_h = len(torch.nonzero(label[k_indices_neg[0]] == label[k_indices_neg[1]])) / len(k_indices_neg[0])
    return homo_r_l, homo_r_h


def normalize_adj_from_tensor(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EPS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EPS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EPS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


args = set_params()
feat, adj, label, labeled_index, idx_train, idx_val, idx_test = \
    load_data(args.dataset)
nb_classes = label.shape[-1]


x = feat
print('x', x.shape)
adj = remove_self_loop([adj])[0].coalesce()

#  adj:

indices = adj.indices()
label_indices = label[indices.flatten()].view(2,-1)
true_indices = (label_indices == 2).sum(dim=0)
true_indices_ = torch.nonzero(true_indices==0).flatten()
indices_ = indices[:,true_indices_]
print(indices_.flatten().unique().shape[0])

label_ill_indices = label[indices_.flatten()].view(2,-1)
ill_indices = (label_ill_indices == 1).sum(dim=0)
ill_indices_ = torch.nonzero(ill_indices!=0).flatten()
ill = indices_[:,ill_indices_]

homo_r = torch.nonzero(label[ill[0]]==label[ill[1]]).flatten().shape[0] / ill.shape[1]
print('adj:',homo_r)
#
# ill_nodes = ill[:,torch.nonzero(label[ill[0]]==label[ill[1]]).flatten()].flatten().unique()
# print(ill_nodes.shape)
# ill_nodes_ = np.intersect1d(ill_nodes.numpy(), torch.nonzero(label==1).flatten().numpy())
# print(ill_nodes_.shape)


# adj_l = torch.load('./lg/' + args.dataset + '_adj_l.pt').cpu().coalesce()
# adj_h = torch.load('./lg/' + args.dataset + '_adj_h.pt').cpu().coalesce()
# pos = torch.load('./lg/' + args.dataset + '_pos.pt').cpu().coalesce()
# adj_l = remove_self_loop([adj_l])[0].coalesce()
# adj_h = remove_self_loop([adj_h])[0].coalesce()
# pos = remove_self_loop([pos])[0].coalesce()
# print(adj._nnz()/len(feat), adj_l._nnz()/len(feat), adj_h._nnz()/len(feat), pos._nnz()/len(feat))
#
# #  adj_l:
#
# indices = adj_l.indices()
# label_indices = label[indices.flatten()].view(2,-1)
# true_indices = (label_indices == 2).sum(dim=0)
# true_indices_ = torch.nonzero(true_indices==0).flatten()
# indices_ = indices[:,true_indices_]
# # print(indices_.flatten().unique().shape[0])
#
# label_ill_indices = label[indices_.flatten()].view(2,-1)
# ill_indices = (label_ill_indices == 1).sum(dim=0)
# ill_indices_ = torch.nonzero(ill_indices!=0).flatten()
# ill = indices_[:,ill_indices_]
#
# homo_r = torch.nonzero(label[ill[0]]==label[ill[1]]).flatten().shape[0] / ill.shape[1]
# print('adj_l:',homo_r)
#
#
# #  adj_h:
#
# indices = adj_h.indices()
# label_indices = label[indices.flatten()].view(2,-1)
# true_indices = (label_indices == 2).sum(dim=0)
# true_indices_ = torch.nonzero(true_indices==0).flatten()
# indices_ = indices[:,true_indices_]
# # print(indices_.flatten().unique().shape[0])
#
# label_ill_indices = label[indices_.flatten()].view(2,-1)
# ill_indices = (label_ill_indices == 1).sum(dim=0)
# ill_indices_ = torch.nonzero(ill_indices!=0).flatten()
# ill = indices_[:,ill_indices_]
#
# homo_r = torch.nonzero(label[ill[0]]==label[ill[1]]).flatten().shape[0] / ill.shape[1]
# print('adj_h:',homo_r)
#
# #  pos:
#
# indices = pos.indices()
# label_indices = label[indices.flatten()].view(2,-1)
# true_indices = (label_indices == 2).sum(dim=0)
# true_indices_ = torch.nonzero(true_indices==0).flatten()
# indices_ = indices[:,true_indices_]
# # print(indices_.flatten().unique().shape[0])
#
# label_ill_indices = label[indices_.flatten()].view(2,-1)
# ill_indices = (label_ill_indices == 1).sum(dim=0)
# ill_indices_ = torch.nonzero(ill_indices!=0).flatten()
# ill = indices_[:,ill_indices_]
#
# homo_r = torch.nonzero(label[ill[0]]==label[ill[1]]).flatten().shape[0] / ill.shape[1]
# print('pos:',homo_r)
#




d = [torch.sum(adj.to_dense(), dim=1) for adj in adjs]
d_zeros = [th.nonzero(dd == 0).flatten().numpy() for dd in d]
for i in range(len(d)):
    d_ = d[i]
    zeros_index = torch.nonzero(d_==0).flatten()
    d_[zeros_index] += EPS
    d[i] = d_

homo_r = [np.zeros(nnodes) for _ in adjs]

for i in range(len(adjs)):
    adj = adjs[i]
    source = adj.indices()[0]
    destination = adj.indices()[1]
    homo_edge_index = th.nonzero(label[source]==label[destination]).flatten()
    print(len(homo_edge_index)/len(source))
    for j in range(len(homo_edge_index)):
        homo_r[i][source[j]] += 1

homo_ratio = [homo_r[i] / d[i] for i in range(len(adjs))]



















# elliptic:
# adj: 0.370452858203415
# 1.1501013402431184 3.0 3.0 2.0
# adj_l: 0.8009283135636927
# adj_h: 0.013304769421580787
# pos: 0.8291510945008008


# T-fin
# adj: 0.30885471395985553
# 1078.4634499580761 3.0 3.0 3.0
# adj_l: 0.6133745434110706
# adj_h: 0.005403158769742311
# pos: 0.6133745434110706