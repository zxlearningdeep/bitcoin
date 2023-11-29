# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .preprocess import *
from .encoder import *
from .loss_fun import *
from torch.autograd import Variable

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha, gamma=2, device='cuda:0', reduction='mean'):

        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss


class LatGRL(nn.Module):
    def __init__(self, feats_dim, hidden_dim, embed_dim, dropout, drop_feat, nnodes, dataset, alpha, nlayer_l, nlayer_h, appnp_alpha):
        super(LatGRL, self).__init__()
        self.alpha = alpha
        self.feats_dim = feats_dim
        self.embed_dim = embed_dim
        self.dataset = dataset
        self.nnodes = nnodes
        self.nlayer_l = nlayer_l
        self.nlayer_h = nlayer_h
        self.appnp_alpha = appnp_alpha

        self.drop = nn.Dropout(drop_feat)

        self.fc_l = nn.Sequential(nn.Linear(feats_dim, hidden_dim),
                                nn.ELU(),
                                nn.BatchNorm1d(hidden_dim),
                                nn.Linear(hidden_dim, embed_dim))
        self.fc_h = nn.Sequential(nn.Linear(feats_dim, hidden_dim),
                                nn.ELU(),
                                nn.BatchNorm1d(hidden_dim),
                                nn.Linear(hidden_dim, embed_dim))


        self.classifier_l = nn.Linear(embed_dim, 2)
        self.classifier_h = nn.Linear(embed_dim, 2)
        self.classifier = nn.Linear(2*embed_dim, 2)

        self.decoder = nn.Sequential(nn.Linear(2*embed_dim, hidden_dim),
                                     nn.ELU(),
                                     nn.BatchNorm1d(hidden_dim),
                                     nn.Linear(hidden_dim, hidden_dim),
                                     nn.ELU(),
                                     nn.BatchNorm1d(hidden_dim),
                                     nn.Linear(hidden_dim, feats_dim),
                                     )
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

        # self.criterion = nn.CrossEntropyLoss()
        self.loss_C = MultiClassFocalLossWithAlpha(alpha=[0.25,0.75]).cuda()

    def forward(self, feat, adj_l, adj_h, label, train_idx, pos_indices):

        # adj_I = torch.eye(self.nnodes, self.nnodes).to(feat.device)

        h_l = self.fc_l(self.drop(feat))
        h_h = self.fc_h(self.drop(feat))

        z_l_ = APPNP(h_l, adj_l, self.nlayer_l, self.appnp_alpha)
        z_h_ = APPNP(h_h, adj_h, self.nlayer_h, 0.0)


        z_l = z_l_
        z_h = z_h_


        z = torch.cat((z_l, z_h), dim=1)

        fea_rec = self.decoder(z)
        loss_rec = sce_loss(fea_rec[pos_indices[0]], feat[pos_indices[1]], self.alpha)
        print(loss_rec.item())


        loss_c = self.loss_C(self.classifier(z)[train_idx], label[train_idx])
        loss_l = self.loss_C(self.classifier_l(z_l)[train_idx], label[train_idx])
        loss_h = self.loss_C(self.classifier_h(z_h)[train_idx], label[train_idx])

        loss = (loss_c+loss_l+loss_h) / 3 + loss_rec

        return loss



    def get_embeds(self, feat, adj_l, adj_h):
        h_l = self.fc_l(feat)
        h_h = self.fc_h(feat)

        z_l_ = APPNP(h_l, adj_l, self.nlayer_l, self.appnp_alpha)
        z_h_ = APPNP(h_h, adj_h, self.nlayer_h, 0.0)


        z_l = z_l_
        z_h = z_h_

        z = torch.cat((z_l, z_h), dim=1)

        return z.detach()

    def test(self, feat, adj_l, adj_h):
        h_l = self.fc_l(feat)
        h_h = self.fc_h(feat)

        z_l_ = APPNP(h_l, adj_l, self.nlayer_l, self.appnp_alpha)
        z_h_ = APPNP(h_h, adj_h, self.nlayer_h, 0.0)


        z_l = z_l_
        z_h = z_h_

        z = torch.cat((z_l, z_h), dim=1)
        pred = self.classifier(z)
        # pred = self.classifier_l(z_l)

        return pred.detach()
