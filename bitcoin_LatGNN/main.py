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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.ensemble import RandomForestClassifier
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.manifold import TSNE


warnings.filterwarnings('ignore')
args = set_params()

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## random seed ##
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_color(labels):
    colors=["b","g","r","y","o"]
    color=[]
    for i in range(len(labels)):
        color.append(colors[labels[i]])
    return color



def train():
    feat, adj, label, labeled_index, idx_train, idx_val, idx_test = \
        load_data(args.dataset)
    nb_classes = 2
    num_target_node = len(feat)

    feats_dim = feat.shape[1]
    print("Dataset: ", args.dataset)
    print("Number of target nodes:", num_target_node)
    print("The dim of target' nodes' feature: ", feats_dim)
    print("Label: ", label.sum(dim=0))
    print(args)

    if torch.cuda.is_available():
        print('Using CUDA')
        # adj = adj.cuda()
        feat = feat.cuda()
        label = label.cuda()
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()

    # adj_l, adj_h, pos = graph_process(adj, feat, args)
    torch.cuda.empty_cache()

    # torch.save(adj_l, './lg/'+args.dataset+'_adj_l.pt')
    # torch.save(adj_h, './lg/'+args.dataset+'_adj_h.pt')
    # torch.save(pos, './lg/'+args.dataset+'_pos.pt')
    adj_l = torch.load('./lg/'+args.dataset+'_adj_l.pt').cuda().coalesce()
    adj_h = torch.load('./lg/'+args.dataset+'_adj_h.pt').cuda().coalesce()
    # pos = torch.load('./lg/'+args.dataset+'_pos.pt').cpu().coalesce().indices().cuda()
    # pos = adj_l.indices()


    model = LatGRL(feats_dim, args.hidden_dim, args.embed_dim, args.dropout, args.drop_feat, len(feat), args.dataset, args.alpha, args.nlayer_l, args.nlayer_h, args.appnp_alpha)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2_coef)
    schedule = MultiStepLR(optimizer, milestones=[800, 2000], gamma=0.4)

    if torch.cuda.is_available():
        model.cuda()

    cnt_wait = 0
    best = 0
    best_pr = 0
    period = 2
    best_epoch=0

    starttime = datetime.datetime.now()

    if args.load_parameters == False:
        for epoch in range(args.nb_epochs):
            model.train()
            optimizer.zero_grad()

            loss = model(feat, adj_l, adj_h, label, idx_train, pos)
            loss.backward()
            optimizer.step()
            schedule.step()


            model.eval()
            embeds = model.test(feat, adj_l, adj_h)[idx_train].cpu()
            logits = F.softmax(embeds, dim=1)
            auroc = roc_auc_score(label[idx_train].cpu(), logits[:, 1])

            lr_precision, lr_recall, _ = precision_recall_curve(label[idx_train].cpu(), logits[:, 1])
            auprc = auc(lr_recall, lr_precision)

            print("Epoch:", epoch, 'train auroc:', auroc, 'train auprc:', auprc)

            if (epoch + 1) % period == 0 and epoch > args.start_eval:
                print("---------------------------------------------------")
                model.eval()
                embeds = model.test(feat, adj_l, adj_h)[idx_test].cpu()
                logits = F.softmax(embeds, dim=1)
                auroc = roc_auc_score(label[idx_test].cpu(), logits[:,1])

                lr_precision, lr_recall, _ = precision_recall_curve(label[idx_test].cpu(), logits[:,1])
                auprc = auc(lr_recall, lr_precision)

                print('test auroc:', auroc, 'test auprc', auprc)

                if best_pr < auprc:
                    best = auroc
                    best_pr = auprc
                    cnt_wait = 0
                    best_epoch = epoch
                    torch.save(model.state_dict(), './checkpoint/' + args.dataset + '/best_' + str(args.seed) + '.pth')
                    print('save model!')
                else:
                    cnt_wait += 1
                if cnt_wait >= args.patience:
                    break
        print('best auroc:', best, 'best auprc:', best_pr)
    else:
        model.load_state_dict(torch.load('./checkpoint/' + args.dataset + '/best_' + str(args.seed) + '.pth'))
        model.eval()
        embeds = model.test(feat, adj_l, adj_h)[idx_test].cpu()
        logits = F.softmax(embeds, dim=1)
        auroc = roc_auc_score(label[idx_test].cpu(), logits[:, 1])

        lr_precision, lr_recall, _ = precision_recall_curve(label[idx_test].cpu(), logits[:, 1])
        auprc = auc(lr_recall, lr_precision)

        print('test auroc:', auroc, 'test auprc', auprc)

        rep = model.get_embeds(feat, adj_l, adj_h).cpu()

        label = label.cpu()
        label_1_index = torch.nonzero(label==1).flatten().unique()
        label_0_index = torch.nonzero(label==0).flatten().unique()
        k_0 = 2000
        k_1 = 1500

        label_1_ = torch.randperm(label_1_index.size(0))
        label_1_index_ = label_1_index[label_1_][:k_1]
        label_0_ = torch.randperm(label_0_index.size(0))
        label_0_index_ = label_0_index[label_0_][:k_0]

        index_ = torch.cat((label_1_index_, label_0_index_),dim=0).unique()
        rep_new = rep[index_]
        row = feat.cpu()[index_]
        label_new = label[index_]

        rep_ = TSNE(n_components=2, init="pca").fit_transform(rep_new)

        figure = plt.figure(figsize=(7, 7), dpi=300)
        color = get_color(label_new)  # 为6个点配置颜色
        x = rep_[:, 0]  # 横坐标
        y = rep_[:, 1]  # 纵坐标
        plt.scatter(x, y, color=color)  # 绘制散点图。
        plt.show()

        row_ = TSNE(n_components=2, init="pca").fit_transform(row)
        figure = plt.figure(figsize=(7, 7), dpi=300)
        color = get_color(label_new)  # 为6个点配置颜色
        x = row_[:, 0]  # 横坐标
        y = row_[:, 1]  # 纵坐标
        plt.scatter(x, y, color=color)  # 绘制散点图。
        plt.show()



if __name__ == '__main__':

    set_seed(args.seed)
    train()