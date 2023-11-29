import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from module.preprocess import remove_self_loop, normalize_adj_from_tensor
import math

EPS = 1e-15


def get_top_k(sim_l, sim_h, k1, k2):
    _, k_indices_pos = torch.topk(sim_l, k=k1, dim=1)
    _, k_indices_neg = torch.topk(sim_h, k=k2, dim=1)

    source = torch.tensor(range(len(sim_l))).reshape(-1, 1).to(sim_l.device)
    k_source_l = source.repeat(1, k1).flatten()
    k_source_h = source.repeat(1, k2).flatten()

    k_indices_pos = k_indices_pos.flatten()
    k_indices_pos = torch.stack((k_source_l, k_indices_pos), dim=0)

    k_indices_neg = k_indices_neg.flatten()
    k_indices_neg = torch.stack((k_source_h, k_indices_neg), dim=0)

    kg_pos = torch.sparse.FloatTensor(k_indices_pos, torch.ones((len(k_indices_pos[0]))).to(sim_l.device), sim_l.shape)
    kg_neg = torch.sparse.FloatTensor(k_indices_neg, torch.ones((len(k_indices_neg[0]))).to(sim_l.device), sim_h.shape)

    return kg_pos, kg_neg


def graph_construction(x, adj, k1, k2, k_pos):
    x = x.cpu()
    adj = adj.cpu().coalesce()
    batchsize = 2000
    batches = math.ceil(len(x) / batchsize)

    adj_ = torch.pow(adj.values(),2)
    adj_ = torch.sparse.FloatTensor(adj.indices(), adj_, adj.shape)
    print(adj_.shape)
    d_ = torch.sparse.sum(adj_, dim=1).values()
    print(d_.shape, x.shape)
    d_ = 1. / (torch.pow(d_, 0.5) + EPS)
    D_value = d_[adj.indices()[0]]
    new_values = adj.values() * D_value
    adj = torch.sparse.FloatTensor(adj.indices(), new_values, adj.size()).coalesce()

    # adj_t = adj.t()

    adj_l_all = []
    adj_h_all = []
    pos_all = []


    for i in range(batches):
        print(i)
        end = (i+1)*batchsize
        if end > len(x):
            end = len(x)
        seed = torch.tensor(range(i*batchsize, end))

        fea_sim = torch.mm(x[seed].cuda(), x.t().cuda())
        adj_seed = torch.index_select(adj, 0, seed).t().to_dense()
        adj_sim = torch.spmm(adj.cuda(), adj_seed.cuda()).t()
        print(fea_sim.shape, adj_sim.shape)

        sim_l = adj_sim * fea_sim
        # sim_l = sim_l - torch.diag_embed(torch.diag(sim_l))
        sim_h = (1 - adj_sim) * (1 - fea_sim)

        kg_pos, kg_neg = get_top_k(sim_l, sim_h, k1+1, k2)

        adj_l_indices = kg_pos._indices()[1].cpu()
        adj_h_indices = kg_neg._indices()[1].cpu()

        pos, _ = get_top_k(sim_l, sim_h, k_pos, k_pos)
        pos = pos._indices()[1].cpu()

        adj_l_all.append(adj_l_indices)
        adj_h_all.append(adj_h_indices)
        pos_all.append(pos)

    adj_l_indices = torch.cat(adj_l_all, dim=0)
    adj_h_indices = torch.cat(adj_h_all, dim=0)
    pos_indices = torch.cat(pos_all, dim=0)


    source = torch.tensor(range(len(x))).reshape(-1, 1)
    source_l = source.repeat(1, k1+1).flatten()
    source_h = source.repeat(1, k2).flatten()
    source_pos = source.repeat(1, k_pos).flatten()



    adj_l = torch.sparse.FloatTensor(torch.stack((source_l, adj_l_indices), dim=0), torch.ones(len(adj_l_indices)), ([len(x), len(x)]))
    adj_h = torch.sparse.FloatTensor(torch.stack((source_h, adj_h_indices), dim=0), torch.ones(len(adj_h_indices)), ([len(x), len(x)]))
    pos = torch.sparse.FloatTensor(torch.stack((source_pos, pos_indices), dim=0), torch.ones(len(pos_indices)), ([len(x), len(x)]))


    return adj_l.coalesce(), adj_h.coalesce(), pos.coalesce()


def graph_process(adj, feat, args):
    adj = adj.coalesce()
    adj = remove_self_loop([adj])[0]  # return sparse tensor
    eye_indices = torch.stack((torch.tensor(range(len(feat))), torch.tensor(range(len(feat)))), dim=0)

    adj = torch.sparse.FloatTensor(torch.cat((eye_indices.cuda(), adj.indices()), dim=1), torch.cat((torch.ones(len(feat)).cuda(), adj.values()), dim=0), ([len(feat), len(feat)])).coalesce()

    adjs_l, adjs_h, pos = graph_construction(feat, adj, args.graph_k, args.graph_k, args.k_pos)

    adjs_l = torch.sparse.FloatTensor(torch.cat((eye_indices, adjs_l.indices()), dim=1), torch.cat((torch.ones(len(feat)), adjs_l.values()), dim=0), ([len(feat), len(feat)])).coalesce().cuda()
    pos = torch.sparse.FloatTensor(torch.cat((eye_indices, pos.indices()), dim=1), torch.cat((torch.ones(len(feat)), pos.values()), dim=0), ([len(feat), len(feat)])).coalesce().cuda()

    adj_l = normalize_adj_from_tensor(adjs_l, mode='row', sparse=True).coalesce()
    adj_h = normalize_adj_from_tensor(adjs_h.cuda(), mode='row', sparse=True).coalesce()
    adj_h_values = -1 * adj_h.values()
    adj_h = torch.sparse.FloatTensor(torch.cat((eye_indices.cuda(), adj_h.indices()), dim=1), torch.cat((torch.ones(len(feat)).cuda(), adj_h_values), dim=0), ([len(feat), len(feat)])).coalesce()

    print(adjs_l)
    print(adjs_h)
    print(pos)


    return adj_l, adj_h, pos



