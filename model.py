import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import GCNConv
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.nn import GINEConv as BaseGINEConv, GINConv as BaseGINConv
from typing import Union, Optional, List, Dict
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, PairTensor


class GraphEmbeddingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphEmbeddingNetwork, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class GINConv(BaseGINConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                edge_atten: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if edge_atten is not None:
            return x_j * edge_atten
        else:
            return x_j


class GINEConv(BaseGINEConv):
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                edge_atten: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_atten=edge_atten, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor, edge_atten: OptTensor = None) -> Tensor:
        if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
            raise ValueError("Node and edge feature dimensionalities do not "
                             "match. Consider setting the 'edge_dim' "
                             "attribute of 'GINEConv'")

        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        m = (x_j + edge_attr).relu()

        if edge_atten is not None:
            return m * edge_atten
        else:
            return m


class GIN(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, n_layers, hidden_size, dropout_p, use_edge_attr,
                 atom_encoder):
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        self.n_layers = n_layers
        hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.use_edge_attr = use_edge_attr

        if atom_encoder:
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool

        for _ in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
            else:
                self.convs.append(GINConv(GIN.MLP(hidden_size, hidden_size)))

        # self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

        self.fc_out = nn.Sequential(GIN.MLP(hidden_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

        '''for CAL'''

        self.c_fc_out = nn.Sequential(GIN.MLP(hidden_size, hidden_size),
                                      nn.BatchNorm1d(hidden_size),
                                      nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))
        self.s_fc_out = nn.Sequential(GIN.MLP(hidden_size, hidden_size),
                                      nn.BatchNorm1d(hidden_size),
                                      nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))
        self.csi_fc_out = nn.Sequential(GIN.MLP(hidden_size, hidden_size),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

        '''for CL'''
        self.projector = GraphEmbeddingNetwork(input_dim=64, output_dim=64)

        '''for RK'''
        self.fc_rk_out = nn.Sequential(GIN.MLP(hidden_size, hidden_size),
                                       nn.BatchNorm1d(hidden_size),
                                       nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.fc_out(self.pool(x, batch))

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))

    def get_pred_from_c_emb(self, emb, batch):
        return self.c_fc_out(self.pool(emb, batch))

    def get_pred_from_s_emb(self, emb, batch):
        return self.s_fc_out(self.pool(emb, batch))

    def get_pred_from_csi_emb(self, emb, batch):
        return self.csi_fc_out(self.pool(emb, batch))

    def get_graph_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.pool(x, batch)

    def get_graph_emb_cl(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.pool(x, batch)
        return self.projector(x)


class GCN(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, n_layers, hidden_size, dropout_p, use_edge_attr,
                 atom_encoder):
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        self.n_layers = n_layers
        hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.use_edge_attr = use_edge_attr

        if atom_encoder:
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_mean_pool

        for _ in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.bns.append(nn.BatchNorm1d(hidden_size))
                self.convs.append(GCNConv(hidden_size, hidden_size))
            else:
                self.bns.append(nn.BatchNorm1d(hidden_size))
                self.convs.append(GCNConv(hidden_size, hidden_size))

        # self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

        self.fc_out = nn.Sequential(GCN.MLP(hidden_size, hidden_size),
                                    nn.BatchNorm1d(hidden_size),
                                    nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

        '''for CAL'''

        self.c_fc_out = nn.Sequential(GCN.MLP(hidden_size, hidden_size),
                                      nn.BatchNorm1d(hidden_size),
                                      nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))
        self.s_fc_out = nn.Sequential(GCN.MLP(hidden_size, hidden_size),
                                      nn.BatchNorm1d(hidden_size),
                                      nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))
        self.csi_fc_out = nn.Sequential(GCN.MLP(hidden_size, hidden_size),
                                        nn.BatchNorm1d(hidden_size),
                                        nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)

        assert edge_attr == None

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_atten)
            x = self.bns[i](x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.fc_out(self.pool(x, batch))

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_atten)
            x = self.bns[i](x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))

    def get_pred_from_c_emb(self, emb, batch):
        return self.c_fc_out(self.pool(emb, batch))

    def get_pred_from_s_emb(self, emb, batch):
        return self.s_fc_out(self.pool(emb, batch))

    def get_pred_from_csi_emb(self, emb, batch):
        return self.csi_fc_out(self.pool(emb, batch))

    def get_graph_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_weight=edge_atten)
            x = self.bns[i](x)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.pool(x, batch)


def get_model(cfg):
    model = None

    if cfg['backbone_name'] == 'gin':
        model = GIN(x_dim=cfg['node_attr_dim'],
                    edge_attr_dim=cfg['edge_attr_dim'],
                    num_class=cfg['num_class'],
                    multi_label=cfg['multi_label'],
                    n_layers=cfg['n_layers'],
                    hidden_size=cfg['hidden_size'],
                    dropout_p=cfg['dropout_p'],
                    use_edge_attr=cfg['use_edge_attr'],
                    atom_encoder=cfg['atom_encoder'])
    elif cfg['backbone_name'] == 'gcn':
        model = GCN(x_dim=cfg['node_attr_dim'],
                    edge_attr_dim=cfg['edge_attr_dim'],
                    num_class=cfg['num_class'],
                    multi_label=cfg['multi_label'],
                    n_layers=cfg['n_layers'],
                    hidden_size=cfg['hidden_size'],
                    dropout_p=cfg['dropout_p'],
                    use_edge_attr=cfg['use_edge_attr'],
                    atom_encoder=cfg['atom_encoder'])

    return model
