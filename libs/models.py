from xml.sax import xmlreader
from .utils import *
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GATConv, GATv2Conv, SAGEConv

class BasicNet(nn.Module):
    def __init__(self, 
                 dimensions : list, 
                 activation = 'tanh',
                 indim : int = 8,
                 outdim : int = 8):
        """
        dimensions: neural network hidden dimensions
        """
        super().__init__()
        in_dims = [indim] + dimensions
        out_dims = dimensions + [outdim]
        layerList = []
        for dims in zip(in_dims, out_dims):
            in_dim, out_dim = dims
            layerList.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layerList)
        self.act = choose_act(activation)
        
    def forward(self, X):
        for _, l in enumerate(self.layers):
            X = self.act(l(X))

        return X

class GAT(torch.nn.Module):
    def __init__(self, 
                 hid : int, 
                 in_head : int, 
                 out_head : int, 
                 in_feature : int = 9, 
                 out_feature : int = 9, 
                 dropout_rate = 0.6,
                 init = 'xavier normal',
                 activation = 'leakyrelu'):

        super(GAT, self).__init__()
        self.hid = hid
        self.in_head = in_head
        self.out_head = out_head
        self.p = dropout_rate
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.act = choose_act(activation)

        self.conv1 = GATv2Conv(self.in_feature, self.hid, heads=self.in_head, dropout=dropout_rate)
        self.conv2 = GATv2Conv(self.hid*self.in_head, self.out_feature, concat=False,
                             heads=self.out_head, dropout=dropout_rate)

        self.init = init
        self.apply(lambda m : init_weights(m, self.init))


    def forward(self, x, edge_index):
        """
        x.shape = (num_nodes, num_features)
        edge_index.shape = (2, num_edges)
        
        x : contains feature info
        edge_index : each column contains the indices of two end nodes of an edge
        """
        shape_b4 = copy.deepcopy(x.shape)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        assert x.shape == shape_b4
        return x

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, dhid, init='xavier uniform', #enforce_symmetry : str = None, 
                 activation=None, ncopy=1, indim=8, outdim=8, dropout=0.1):

        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert dhid % h == 0
        # We assume d_v always equals d_k
        self.d_k = dhid // h
        self.h = h
        self.encoder = nn.Linear(indim, dhid)
        self.linears = clones(nn.Linear(dhid, dhid), 4)
        self.decoder = nn.Linear(dhid, outdim)
        self.attn = None
        self.act = choose_act(activation)
        self.init = init
        self.dropout = nn.Dropout(p=dropout)
        self.ncopy = ncopy
        # self.symmetry = enforce_symmetry
        self.ID = nn.Identity
        self.apply(lambda m : init_weights(m, self.init))
        
    def forward(self, x, mask=None):
        """
        x : input stencil
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        assert x.size(0) == 1 # batch is 1
        x = x.repeat(self.ncopy,1)
        x = self.encoder(x)
        x = self.act(x)
        query, key, value = x, x, x
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        value = self.act(value)
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value,
                                 mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        x = self.decoder(self.linears[-1](x))
        x = x.mean(dim=0)
        # print(x.shape)

        ##### TODO: identity block??
        # assert 0==1
        # if self.symmetry is not None:
        #     # print(x)
        #     identity = self.ID(x)
        #     x = identity + torch.flip(x,[0])
        #     # print(x)
        #     # assert 0==1

        return x

class MultiHeadedAttention2(nn.Module):
    def __init__(self, h, dhid, init='xavier uniform', indim=8, outdim=8, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention2, self).__init__()
        assert dhid % h == 0
        # We assume d_v always equals d_k
        self.d_k = dhid // h
        self.h = h
        self.indim = indim
        self.outdim = outdim
        self.encoder = nn.Linear(indim, dhid)
        self.linears = clones(nn.Linear(dhid, dhid), 4)
        self.decoder = nn.Linear(dhid, outdim)
        self.attn = None
        self.init = init
        self.dropout = nn.Dropout(p=dropout)
        self.apply(lambda m : init_weights(m, self.init))
        
    def forward(self, x, mask=None):
        """
        x : input stencil
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        assert x.size(1) == self.indim 
        x = self.encoder(x)
        query, key, value = x, x, x
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value,
                                 mask=mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        x = self.decoder(self.linears[-1](x))
        return x

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def sparsify(prob,value):
    stencil = top_k(prob,4).squeeze()
    stencil = stencil*value
    # print(stencil)
    stencil = torch.cat((stencil[0:4],-stencil.sum().view(1),stencil[4:8])).view(1,1,3,3)
    return stencil

def choose_act(activation):
    if activation is None:
        return lambda x: x
    elif activation == 'relu':
        return lambda x: F.relu(x, inplace=False)
    elif activation == 'gelu':
        return F.gelu
    elif activation == 'tanh':
        return F.tanh
    elif activation == 'elu':
        return F.elu
    elif activation == 'leakyrelu':
        return lambda x: F.leaky_relu(x, inplace=False)

def init_weights(m, init):
    initialize = choose_init(init)
    if isinstance(m, nn.Linear):
        initialize(m.weight)
        m.bias.data.fill_(0.01)

def choose_init(init:str):
    if init=='xavier uniform':
        return torch.nn.init.xavier_uniform_
    elif init == 'xavier normal':
        return torch.nn.init.xavier_normal_
    elif init == 'kaiming uniform':
        return torch.nn.init.kaiming_uniform_
    elif init == 'kaiming normal':
        return torch.nn.init.kaiming_normal_
    else:
        raise ValueError(f"{init} is not an available initialization")

def choose_conv(conv:str):
    if conv == 'GAT':
        return 