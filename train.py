from torch.optim import Adam
from torch import tensor
from utils import *
from tqdm import tqdm
from torch_geometric.utils.convert import to_networkx
import time
import networkx as nx
from os import path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(data, alpha, beta, model, optimizer, graph_learner, args):
    model.train()
    optimizer.zero_grad()
    x, edge_index = data.x, data.edge_index
    x_bar, Z, x_aug, output = model(data.x)

    if args.augmentation == 'knn':
        knn_edge_index = KNN_graph(x, k=args.k)
        hs1 = model.prop(x_aug, knn_edge_index)
    elif args.augmentation == 'drop':
        drop_edge_index = edge_drop(edge_index, p=args.edge_drop)
        hs1 = model.prop(x_aug, drop_edge_index)
    elif args.augmentation == 'init':
        hs1 = model.prop(x_aug, edge_index)
    elif args.augmentation == 'MIP':
        hs1 = model.prop_mip(x_aug, args.hop_edge_index, args.hop_edge_att)
    else:
        raise ValueError("false")

    # sup_loss
    loss_train_class = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss_train = loss_train_class + alpha * model.AMNC_sum(Z, hs1) + beta * F.mse_loss(x_bar, data.x)
    loss_train.backward()
    optimizer.step()
    return


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data.x)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs


def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping, alpha, beta, graph_learner, permute_masks, args, lcc=False):
    val_losses, accs, durations = [], [], []

    lcc_mask = None
    if lcc:  # select largest connected component
        data_ori = dataset[0]
        data_nx = to_networkx(data_ori)
        data_nx = data_nx.to_undirected()
        data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
        lcc_mask = list(data_nx.nodes)

    pbar = tqdm(range(runs), unit='run')
    data = dataset[0]

    # run multi_hop decomposition
    edge_file = './multi_info/hop_edge_index_' + args.dataset + '_' + str(args.multi_layer)
    if (path.exists(edge_file) == False):
        edge_info(dataset, args)
    # load multi_hop decomposed edge_index and multi-hop edge weight(att)
    hop_edge_index = torch.load('./multi_info/hop_edge_index_' + args.dataset + '_' + str(args.multi_layer))
    hop_edge_att = torch.load('./multi_info/hop_edge_att_' + args.dataset + '_' + str(args.multi_layer))
    for _ in pbar:
        if permute_masks is not None:
            train_rate = 0.6
            val_rate = 0.2
            percls_trn = int(round(train_rate * len(data.y) / dataset.num_classes))
            val_lb = int(round(val_rate * len(data.y)))
            data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)
        data = data.to(device)
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        # get_multi_layer
        for layer in args.layers:
            hop_edge_index[layer - 1] = hop_edge_index[layer - 1].type(torch.LongTensor).to(device)
            hop_edge_att[layer - 1] = hop_edge_att[layer - 1].to(device)
        args.hop_edge_index = hop_edge_index
        args.hop_edge_att = hop_edge_att

        for epoch in range(1, epochs + 1):
            out = train(data, alpha, beta, model, optimizer, graph_learner, args)
            eval_info = evaluate(model, data)
            eval_info['epoch'] = epoch
            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']
            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)
    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)
    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))
