import numpy as np
import torch
from utils import load_data, set_params
from module.LatGRL import *
from module.graph_generating import *
from module.preprocess import *
import warnings
import datetime
import pickle as pkl
import random
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc

warnings.filterwarnings('ignore')
args = set_params()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


feat, adj, label, labeled_index, idx_train, idx_val, idx_test = \
    load_data(args.dataset)
nb_classes = 2
num_target_node = len(feat)

feats_dim = feat.shape[1]
print("Dataset: ", args.dataset)
print("Number of target nodes:", num_target_node)
print("The dim of target' nodes' feature: ", feats_dim)
print(args)

adj_l = torch.load('./lg/' + args.dataset + '_adj_l.pt').cuda()
adj_h = torch.load('./lg/' + args.dataset + '_adj_h.pt').cuda()
# pos = torch.load('./lg/' + args.dataset + '_pos.pt').cpu()
feat = feat.cuda()

z_l_1 = APPNP(feat, adj_l, 1, 0.4).cpu()
z_h_1 = APPNP(feat, adj_h, 1, 0.0).cpu()
z_l_2 = APPNP(feat, adj_l, 2, 0.4).cpu()
z_h_2 = APPNP(feat, adj_h, 2, 0.0).cpu()
feat = feat.cpu()
z = torch.cat((feat, z_l_1, z_l_2, z_h_1, z_h_2), dim=1)

rfc = RandomForestClassifier(n_estimators=150, random_state=2)
rfc.fit(z[idx_train], label[idx_train])
test_prob = rfc.predict_proba(z[idx_test])[:,1]

auroc = roc_auc_score(label[idx_test].cpu(), test_prob)

lr_precision, lr_recall, _ = precision_recall_curve(label[idx_test].cpu(),test_prob)
auprc = auc(lr_recall, lr_precision)

print('test auroc:', auroc, 'test auprc', auprc)