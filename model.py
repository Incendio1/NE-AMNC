import torch.nn as nn
from torch.nn import Dropout, Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from utils import *


class GCN_prop(MessagePassing):
    def __init__(self, K, args, **kwargs):
        super().__init__(aggr='add', **kwargs)
        self.weight = torch.nn.Parameter(torch.ones(K+1), requires_grad=True)
        self.K = K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        reps = []
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            reps.append(x)
        return reps


class MIP_prop(MessagePassing):
    def __init__(self, args, **kwargs):
        super(MIP_prop, self).__init__(aggr='add', **kwargs)
        self.layers = args.layers

    def forward(self, x, edge_index, edge_weight):
        embed_layer = []
        if(self.layers != [0]):
            for layer in self.layers:
                x = self.propagate(edge_index[layer - 1], x=x, norm=edge_weight[layer - 1])
                embed_layer.append(x)
        return embed_layer


class AE(nn.Module):

    def __init__(self, n_hidden, n_input, n_z, dropout):
        super(AE, self).__init__()
        self.dropout = dropout
        self.enc_1 = Linear(n_input, n_hidden)
        self.z_layer = Linear(n_hidden, n_z)
        self.dec_1 = Linear(n_z, n_hidden)
        self.x_bar_layer = Linear(n_hidden, n_input)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.enc_1.weight)
        nn.init.xavier_uniform_(self.z_layer.weight)
        nn.init.xavier_uniform_(self.dec_1.weight)
        nn.init.xavier_uniform_(self.x_bar_layer.weight)
        nn.init.normal_(self.enc_1.bias, std=1e-6)
        nn.init.normal_(self.z_layer.bias, std=1e-6)
        nn.init.normal_(self.dec_1.bias, std=1e-6)
        nn.init.normal_(self.x_bar_layer.bias, std=1e-6)

    def reset_parameters(self):
        self.enc_1.reset_parameters()
        self.z_layer.reset_parameters()
        self.dec_1.reset_parameters()
        self.x_bar_layer.reset_parameters()


    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h1 = F.dropout(enc_h1, p=self.dropout, training=self.training)

        z = self.z_layer(enc_h1)
        z_drop = F.dropout(z, p=self.dropout, training=self.training)

        dec_h1 = F.relu(self.dec_1(z_drop))
        dec_h1 = F.dropout(dec_h1, p=self.dropout, training=self.training)

        x_bar = self.x_bar_layer(dec_h1)

        return x_bar, z


class NE_AMNC(nn.Module):
    def __init__(self, nhid, n_z, nfeat, nclass, dropout, args):
        super(NE_AMNC, self).__init__()
        self.K = args.K
        if args.Init == 'random':
            # random
            bound = np.sqrt(3/(self.K))
            logits = np.random.uniform(-bound, bound, self.K)
            logits = logits/np.sum(np.abs(logits))
            self.logits = Parameter(torch.tensor(logits))
            print(f"init logits: {logits}")
        else:
            logits = np.array([1, float('-inf'), float('-inf')])
            self.logits = torch.tensor(logits)

        self.AE = AE(nhid, nfeat, n_z, dropout)
        self.classifier = Linear(n_z, nclass)
        self.prop = GCN_prop(self.K, args)
        self.prop_mip = MIP_prop(args)

    def forward(self, x):
        x_bar, x = self.AE(x)
        x_augment = rand_prop(x, training=self.training)
        class_feature = self.classifier(x)
        class_logits = F.log_softmax(class_feature, dim=1)

        if self.training:
            return x_bar, x, x_augment, class_logits
        else:
            return class_logits

    def reset_parameters(self):
        self.AE.reset_parameters()
        self.classifier.reset_parameters()
        torch.nn.init.zeros_(self.logits)
        bound = np.sqrt(3/(self.K))
        logits = np.random.uniform(-bound, bound, self.K)
        logits = logits/np.sum(np.abs(logits))
        for k in range(self.K):
            self.logits.data[k] = logits[k]

        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def AMNC(self, h1, h2, gamma, temperature=1.0, bias=1e-8):
        # h1: x, h2: n-hop neighbors
        z1 = F.normalize(h1, dim=-1, p=2)
        z2 = gamma*F.normalize(h2, dim=-1, p=2)
        numerator = torch.exp(
            torch.sum(z1 * z2, dim=1, keepdims=True) / temperature)

        E_1 = torch.matmul(z1, torch.transpose(z1, 1, 0))

        denominator = torch.sum(
            torch.exp(E_1 / temperature), dim=1, keepdims=True)

        return -torch.mean(torch.log(numerator / (denominator + bias) + bias))

    def AMNC_sum(self, h0, hs):
        # h0: x; hs: list of h1, h2 ...hk
        loss = torch.tensor(0, dtype=torch.float32).cuda()
        gamma = F.softmax(self.logits, dim=0)
        for i in range(len(hs)):
            loss += self.AMNC(h0, hs[i], gamma[i])

        return loss