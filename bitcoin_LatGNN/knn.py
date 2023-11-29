import numpy as np
import torch
import torch as th
from torch import Tensor
import warnings


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from utils.load_data import load_data
from utils.params import set_params
from module.preprocess import remove_self_loop
warnings.filterwarnings('ignore')

EPS = 1e-15



def find_idx(a: Tensor, b: Tensor, missing_values: int = -1):
    """Find the first index of b in a, return tensor like a."""
    a, b = a.clone(), b.clone()
    invalid = ~torch.isin(a, b)
    a[invalid] = b[0]
    sorter = torch.argsort(b)
    b_to_a: Tensor = sorter[torch.searchsorted(b, a, sorter=sorter)]
    b_to_a[invalid] = missing_values
    return b_to_a





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


x = feat.cuda()
label = label.cuda()
print('x', x.shape)
adj = remove_self_loop([adj])[0].coalesce().cuda()



# indices = adj.indices()
# ill = torch.nonzero(label==1).flatten().unique()
# label_ill = label[ill]
# adj_ill = torch.index_select(adj, 0, ill).coalesce()
#
#
# d = torch.sum(adj_ill.to_dense(), dim=1)
# d_zeros = th.nonzero(d == 0).flatten()
# d[d_zeros] += EPS
#
# homo_r = torch.zeros(len(ill)).cuda()
# source = adj_ill.indices()[0]
# destination = adj_ill.indices()[1]
# homo_edge_index = th.nonzero(label_ill[source] == label[destination]).flatten()
# print(len(homo_edge_index) / len(source))
# for j in range(len(homo_edge_index)):
#     homo_r[source[j]] += 1
#
# homo_ratio = homo_r / d
#
# print(homo_ratio)




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
indices_ = ill



indices_unique = indices_.flatten().unique()
adj_indices = torch.index_select(adj, 0, indices_unique).coalesce()
adj_indices = torch.index_select(adj_indices, 1, indices_unique).coalesce()
label_indices = label[indices_unique]
indices_in = find_idx(indices_.flatten(), indices_unique).view(2,-1)









#
# d = torch.sum(adj_indices.to_dense(), dim=1)
# d_zeros = th.nonzero(d == 0).flatten()
# d[d_zeros] += EPS
#
# homo_r = torch.zeros(len(indices_unique)).cuda()
# source = adj_indices.indices()[0]
# destination = adj_indices.indices()[1]
# homo_edge_index = th.nonzero(label_indices[source] == label_indices[destination]).flatten()
# print(len(homo_edge_index) / len(source))
# for j in range(len(homo_edge_index)):
#     homo_r[source[j]] += 1
#
# homo_ratio = homo_r / d
#
# print(homo_ratio)
















#
# homo_ratio = homo_ratio.cpu()
# # yelp 1d Draw Plot
# plt.figure(figsize=(10, 7), dpi=180)
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams.update({'font.size':15})
# plt.title('Density of Homophily Ratio (1D)', fontsize=25, fontweight='bold')
#
# # sns.kdeplot(homo_ratio[0].numpy(), fill=True, color="g", label="homo ratio", alpha=.7)
# sns.kdeplot(homo_ratio.numpy(), fill=True, color="g", alpha=0.8)
#
# # darkseagreen
# plt.legend(loc='upper right')
#
# plt.xlim(0,1)
# # plt.ylim(0,1)
#
# plt.xlabel('Node-Level Homophily Ratio',fontsize=20,fontweight='bold')
# plt.ylabel('Density',fontsize=20,fontweight='bold')
# plt.yticks(fontproperties='Times New Roman', size=15, weight='bold')
# plt.xticks(fontproperties='Times New Roman', size=15, weight='bold')
#
# plt.savefig('./fig/elliptic_hr.svg', dpi=600, format='svg')
# plt.show()
#



