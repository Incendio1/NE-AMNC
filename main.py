import argparse
import numpy as np
from train import *
from model import NE_AMNC
from utils import *
from datasets import *
import warnings
from graph_learner import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--hidden_z', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=False)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--tau', type=float, default=0.5)
parser.add_argument('--K', type=int, default=3, help="number of layer in AMNCLoss")
parser.add_argument('--Init', type=str, default='random',
                    help='the init method gamma logits')
parser.add_argument('--k', type=int, default=8, help="number of neighbors of knn augmentation")
parser.add_argument('--augmentation', type=str, default='MIP',
                    choices=['knn', 'init', 'drop', 'MIP'])
parser.add_argument('--edge_drop', type=float, default=0.2)
parser.add_argument('--knn_metric', type=str,
                    default='cosine', choices=['cosine', 'minkowski'])
parser.add_argument('-activation_learner', type=str,
                    default='relu', choices=["relu", "tanh"])
parser.add_argument('--sparse', action='store_true', default=False)
parser.add_argument('--ppr_rate', type=float, default=0.1)
parser.add_argument('--feat_drop', type=float, default=0.2)
parser.add_argument('--layers', nargs='+', type=int)
parser.add_argument('--multi_layer', type=int, default=10)
args = parser.parse_args()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


if args.dataset == "Cora" or args.dataset == "Citeseer" or args.dataset == "Pubmed":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits_1
    print(f'Dataset:{args.dataset}')
    learner = learner(2, isize=dataset.num_features, k=args.k, knn_metric=args.knn_metric,
                                i=6, sparse=args.sparse, mlp_act=args.activation_learner)

    learner = learner.to(device)
    model = NE_AMNC(args.hidden, args.hidden_z, dataset.num_features, dataset.num_classes, args.dropout, args)
    run(dataset, model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping,
         args.alpha, args.beta, learner,  permute_masks, args, lcc=False)

elif args.dataset == "Wisconsin" or args.dataset == "Cornell" or args.dataset == "Texas":
    dataset = get_WebKB_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits_1
    print(f'Dataset:{args.dataset}')
    print("Dataset:", dataset[0])
    graph_learner = learner(2, isize=dataset.num_features, k=args.k, knn_metric=args.knn_metric,
                                i=6, sparse=args.sparse, mlp_act=args.activation_learner)

    graph_learner = graph_learner.to(device)
    model = NE_AMNC(args.hidden, args.hidden_z, dataset.num_features, dataset.num_classes, args.dropout, args)
    run(dataset, model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping,
        args.alpha, args.beta, graph_learner, permute_masks, args, lcc=False)


elif args.dataset == "Actor":
    dataset = get_Actor_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits_1
    print(f'Dataset:{args.dataset}')
    # print("Dataset:", dataset[0])
    graph_learner = learner(2, isize=dataset.num_features, k=args.k, knn_metric=args.knn_metric,
                                i=6, sparse=args.sparse, mlp_act=args.activation_learner)

    graph_learner = graph_learner.to(device)
    model = NE_AMNC(args.hidden, args.hidden_z, dataset.num_features, dataset.num_classes, args.dropout, args)
    run(dataset, model, args.runs, args.epochs, args.lr, args.weight_decay, args.early_stopping,
        args.alpha, args.beta, graph_learner, permute_masks, args, lcc=False)


