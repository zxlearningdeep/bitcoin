import numpy as np
import scipy.sparse as sp
import torch
import torch as th
from sklearn.preprocessing import OneHotEncoder
# from torch_geometric.datasets import EllipticBitcoinDataset
# from dgl.data.utils import load_graphs
from sklearn.model_selection import train_test_split

EPS = 1e-15

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def adj_values_one(adj):
    adj = adj.coalesce()
    index = adj.indices()
    return th.sparse.FloatTensor(index, th.ones(len(index[0])), adj.shape)

def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)

    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def sp_tensor_to_sp_csr(adj):
    adj = adj.coalesce()
    row = adj.indices()[0]
    col = adj.indices()[1]
    data = adj.values()
    shape = adj.size()
    adj = sp.csr_matrix((data, (row, col)), shape=shape)
    return adj


def load_elliptic():
    path = './data/elliptic/'
    features = torch.load(path + 'features.pt')

    x_norm = torch.norm(features, dim=-1, p=2, keepdim=True)
    zero_indices = torch.nonzero(x_norm.flatten() == 0)
    x_norm[zero_indices] += EPS
    x = features / x_norm

    adj = torch.load(path + 'adj.pt').coalesce()
    y = torch.load(path + 'label.pt')
    labeled_index = torch.load(path+'labeled_index.pt')
    train_mask = torch.load(path + 'train_mask.pt')
    test_mask = torch.load(path + 'test_mask.pt')

    eye_indices = torch.stack((torch.tensor(range(len(features))), torch.tensor(range(len(features)))), dim=0)
    adj = torch.sparse.FloatTensor(torch.cat((eye_indices, adj.indices()), dim=1), torch.cat((torch.ones(len(features)), adj.values()), dim=0), ([len(features), len(features)])).coalesce()



    return x, adj, y, labeled_index, train_mask, test_mask, test_mask


def load_T_fin():
    path = './data/t_fin/'
    features = torch.load(path+'features.pt')

    x = features

    adj = torch.load(path+'adj.pt').coalesce()
    label = torch.load(path+'label.pt')
    index = torch.tensor(range(len(label)))
    idx_train = torch.load(path+'idx_train.pt')
    idx_test= torch.load(path+'idx_test.pt')
    idx_train = torch.tensor(idx_train).long()
    idx_test = torch.tensor(idx_test).long()

    eye_indices = torch.stack((torch.tensor(range(len(features))), torch.tensor(range(len(features)))), dim=0)
    adj = torch.sparse.FloatTensor(torch.cat((eye_indices, adj.indices()), dim=1), torch.cat((torch.ones(len(features)), adj.values()), dim=0), ([len(features), len(features)])).coalesce()



    return x, adj, label, index, idx_train, idx_test, idx_test


def load_data(dataset):
    if dataset == "elliptic":
        data = load_elliptic()
    elif dataset == "T_fin":
        data = load_T_fin()
    return data

