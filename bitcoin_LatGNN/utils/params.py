import argparse
import torch
import sys

# argv = sys.argv
# dataset = argv[1]
# dataset = 'T_fin'
dataset = 'elliptic'

def elliptic_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="elliptic")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=500)  # 400
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=0)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer_l', type=int, default=2)
    parser.add_argument('--nlayer_h', type=int, default=2)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=1e-3)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=200)
    parser.add_argument('--l2_coef', type=float, default=5e-5)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--drop_feat', type=float, default=0.1)

    # model-specific parameters
    parser.add_argument('--appnp_alpha', type=float, default=0.4)
    parser.add_argument('--graph_k', type=int, default=3)
    parser.add_argument('--k_pos', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=1)

    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    return args


def T_fin_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--turn', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="T_fin")
    parser.add_argument('--ratio', type=int, default=[20, 40, 60])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=3000)  # 400
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--auc_limit', type=int, default=0)
    parser.add_argument('--act', default=torch.nn.ELU())
    parser.add_argument('--nlayer_l', type=int, default=2)
    parser.add_argument('--nlayer_h', type=int, default=2)

    # The parameters of evaluation
    parser.add_argument('--eva_lr', type=float, default=0.05)
    parser.add_argument('--eva_wd', type=float, default=1e-3)

    # The parameters of learning process
    parser.add_argument('--patience', type=int, default=500)
    parser.add_argument('--l2_coef', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--drop_feat', type=float, default=0.)

    # model-specific parameters
    parser.add_argument('--appnp_alpha', type=float, default=0.2)
    parser.add_argument('--graph_k', type=int, default=3)
    parser.add_argument('--k_pos', type=int, default=3)
    parser.add_argument('--alpha', type=float, default=0)

    args, _ = parser.parse_known_args()
    args.type_num = [4019, 7167, 60]  # the number of every node type
    return args



def set_params():
    if dataset == "elliptic":
        args = elliptic_params()
    elif dataset == "T_fin":
        args = T_fin_params()

    return args

