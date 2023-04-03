import numpy as np
import scipy.sparse as sp
import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
import copy
import math
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_scatter import scatter


def index_to_mask_1(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_planetoid_splits_1(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag is 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask_1(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask_1(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask_1(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask_1(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask_1(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask_1(rest_index, size=data.num_nodes)
    return data

# Get feature similarity Matrix
def get_feature_dis(x):
    """
    x :           batch_size x nhid
    x_dis(i,j):   item means the similarity between x(i) and x(j).
    """
    x_dis = x @ x.T
    mask = torch.eye(x_dis.shape[0]).cuda()
    x_sum = torch.sum(x ** 2, 1).reshape(-1, 1)
    x_sum = torch.sqrt(x_sum).reshape(-1, 1)
    x_sum = x_sum @ x_sum.T
    x_dis = x_dis * (x_sum ** (-1))
    x_dis = (1 - mask) * x_dis
    return x_dis


#
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


#
def get_batch(batch_size, adj_label, idx_train, features):
    """
    get a batch of feature & adjacency matrix
    """
    rand_indx = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).type(torch.long).cuda()
    rand_indx[0:len(idx_train)] = idx_train
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx, :][:, rand_indx]
    return features_batch, adj_label_batch


# Neighbor Contrast Loss
def Ncontrast(x_dis, adj_label, tau=1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def random_coauthor_amazon_splits(data, num_classes, lcc_mask):
    # Set random coauthor/co-purchase splits:
    # * 20 * num_classes labels for training
    # * 30 * num_classes labels for validation
    # rest labels for testing

    indices = []
    if lcc_mask is not None:
        for i in range(num_classes):
            index = (data.y[lcc_mask] == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)
    else:
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)
    val_index = torch.cat([i[20:50] for i in indices], dim=0)

    rest_index = torch.cat([i[50:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(val_index, size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index, size=data.num_nodes)

    return data


def get_adj_ori(data):
    G = to_networkx(data)
    A = nx.convert_matrix.to_numpy_array(G)
    return A


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def get_A_hat_k_power(sp_adj, k):
    sp_adj_k = sp_adj
    for i in range(k - 1):
        sp_adj_k = torch.sparse.mm(sp_adj_k, sp_adj)
    return sp_adj_k


def get_ehanced_A_hat_k_power(sp_adj, k):
    sp_adj_k = sp_adj
    enhanced_adj = sp_adj
    adj_arr = []
    for i in range(k - 1):
        sp_adj_k = torch.sparse.mm(sp_adj_k, sp_adj)
        adj_arr.append(sp_adj_k)
        enhanced_adj = enhanced_adj + adj_arr[i]

    return enhanced_adj


def KNN_graph(x, k=12):
    # KNN-graph
    h = F.normalize(x, dim=-1)
    device = x.device
    logits = torch.matmul(h, h.t())
    _, indices = torch.topk(logits, k=k, dim=-1)
    graph = torch.zeros(h.shape[0], h.shape[0], dtype=torch.int64, device=device).scatter_(1, indices, 1)

    edge_index = torch.nonzero(graph).t()
    edge_index = to_undirected(edge_index)

    return edge_index


def edge_drop(edge_index, p=0.4):
    # copy edge_index
    edge_index = copy.deepcopy(edge_index)
    num_edges = edge_index.size(1)
    num_droped = int(num_edges * p)
    perm = torch.randperm(num_edges)

    edge_index = edge_index[:, perm[:num_edges - num_droped]]

    return edge_index


def rand_prop(features, training):  # Mask_Node
    n = features.shape[0]
    drop_rate = 0.0
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)

    if training:

        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        features = masks.cuda() * features

    else:

        features = features * (1. - drop_rate)
    # features = propagate(features, A, args.order)
    return features


def edgelist2graph(edge_index, nodenum):
    """ Preprocess to get the graph data wanted
    """
    edge_index = edge_index.cpu().detach().numpy()
    adjlist = {i: [] for i in range(nodenum)}

    for i in range(len(edge_index[0])):
        adjlist[edge_index[0][i]].append(edge_index[1][i])


    return adjlist, nx.adjacency_matrix(nx.from_dict_of_lists(adjlist)).toarray()


def edge_info(dataset, args):
    adjlist, adjmatrix = edgelist2graph(dataset.data.edge_index, dataset.data.x.size(0))
    hop_edge_index, hop_edge_att = multi_decomposition(dataset.data.x.size(0), args.multi_layer, adjlist, dataset.data.edge_index)
    torch.save(hop_edge_index, './multi_info/hop_edge_index_' + args.dataset + '_' + str(args.multi_layer))
    torch.save(hop_edge_att, './multi_info/hop_edge_att_' + args.dataset + '_' + str(args.multi_layer))


def propagate(x, edge_index):
    """ feature propagation procedure: sparsematrix
    """
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes = x.size(0))

    #calculate the degree normalize term
    row, col = edge_index
    deg = degree(col, x.size(0), dtype = x.dtype)
    deg_inv_sqrt = deg.pow(-0.5)
    edge_weight = deg_inv_sqrt[row]*deg_inv_sqrt[col]    # for the first order appro of laplacian matrix in GCN, we use deg_inv_sqrt[row]*deg_inv_sqrt[col]

    out = edge_weight.view(-1, 1)*x[row]    # normalize the features on the starting point of the edge

    return scatter(out, edge_index[-1], dim=0, dim_size=x.size(0), reduce='add')


def weight_deg(nodenum, edge_index, K):
    att = []
    x = torch.eye(nodenum)    # E
    for i in range(K):
        x = propagate(x, edge_index)    # E*A=A
        att.append(x)

    return att


def multi_decomposition(nodenum, K, adjlist, edge_index):
    # calculate the attention, the weight of each edge
    att = weight_deg(nodenum, edge_index, K)

    # to save the space, we use torch.tensor instead of list to save the edge_index
    hop_edge_index = [np.zeros((2, nodenum**2), dtype=int) for i in range(K)]    # at most nodenum**2 edges
    hop_edge_att = [np.zeros(nodenum**2) for i in range(K)]
    hop_edge_pointer = np.zeros(K, dtype=int)

    for i in range(nodenum):
        hop_edge_index, hop_edge_att, hop_edge_pointer = BFS_adjlist(adjlist, nodenum, hop_edge_index, hop_edge_att, hop_edge_pointer, att, source=i, depth_limit=K)

    for i in range(K):
        hop_edge_index[i] = hop_edge_index[i][:, :hop_edge_pointer[i]]
        hop_edge_att[i] = hop_edge_att[i][:hop_edge_pointer[i]]

        hop_edge_index[i] = torch.tensor(hop_edge_index[i], dtype=torch.long)
        hop_edge_att[i] = torch.tensor(hop_edge_att[i], dtype=torch.float)

    return hop_edge_index, hop_edge_att


def BFS_adjlist(adjlist, nodenum, hop_edge_index, hop_edge_att, hop_edge_pointer, deg_att, source, depth_limit):
    visited = {}
    for node in adjlist.keys():
        visited[node] = 0

    queue, output = [], []
    queue.append(source)
    visited[source] = 1
    level = 1

    # initialize the edge pointed to the source node itself
    for i in range(len(hop_edge_index)):
        hop_edge_index[i][0, hop_edge_pointer[i]] = source
        hop_edge_index[i][1, hop_edge_pointer[i]] = source

    tmp = 0
    for k in range(0, depth_limit):
        tmp += deg_att[k][source, source]
    hop_edge_att[0][hop_edge_pointer[0]] = tmp
    hop_edge_pointer[0] += 1

    while queue:
        level_size = len(queue)
        while(level_size != 0):
            vertex = queue.pop(0)
            level_size -= 1
            for vrtx in adjlist[vertex]:
                if(visited[vrtx] == 0):
                    queue.append(vrtx)
                    visited[vrtx] = 1

                    hop_edge_index[level - 1][0, hop_edge_pointer[level - 1]] = source    # distance = 1 is the first group in the hop_edge_list
                    hop_edge_index[level - 1][1, hop_edge_pointer[level - 1]] = vrtx
                    tmp = 0
                    for k in range((level - 1), depth_limit):
                        tmp += deg_att[k][source, vrtx]
                    hop_edge_att[level - 1][hop_edge_pointer[level - 1]] = tmp
                    hop_edge_pointer[level - 1] += 1

        level += 1
        if (level > depth_limit):
            break

    return hop_edge_index, hop_edge_att, hop_edge_pointer



