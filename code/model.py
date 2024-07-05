from os import XATTR_SIZE_MAX
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
import torch_geometric.nn as geo_nn
from torch_geometric.nn.conv.gin_conv import GINEConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv as gcn, DenseGCNConv as GCNConv, DenseGATConv as GATConv, GraphSAGE
from torch_geometric.nn import DenseGINConv as GINConv, DenseSAGEConv as SAGEConv
from torch_geometric.nn import dense_diff_pool, TopKPooling
from torch_geometric.utils import dense_to_sparse, to_dense_adj, add_self_loops
from torch_geometric.utils.dropout import dropout_edge
from torch_geometric.data import Data, Batch


from CTPooling import ct_pool

class New_GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True, norm_channels=None):
        super(New_GNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(gcn(in_channels, hidden_channels))
        self.convs.append(gcn(hidden_channels, hidden_channels))
        self.convs.append(gcn(hidden_channels, out_channels))


    def forward(self, x, adj_index, adj_weight):
        
        for step in range(len(self.convs)-1):

            x = F.relu(self.convs[step](x, adj_index, adj_weight))
        
        x = F.dropout(x, p=0.5)
        x = F.relu(self.convs[step+1](x, adj_index, adj_weight))

        return x

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True, norm_channels=None):
        super(GNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(norm_channels))
        
        # self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        # self.bns.append(torch.nn.BatchNorm1d(norm_channels))
        
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(norm_channels))

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        for step in range(len(self.convs)-1):
            x = self.bns[step](F.relu(self.convs[step](x, adj, mask)))

        # x = F.dropout(x, p=0.5)
        x = self.bns[step+1](F.relu(self.convs[step+1](x, adj, mask)))

        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                norm_channels=None):
        super(GAT, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GATConv(in_channels, hidden_channels))
        self.bns.append(torch.nn.BatchNorm1d(norm_channels))
        
        # self.convs.append(GATConv(hidden_channels, hidden_channels))
        # self.bns.append(torch.nn.BatchNorm1d(norm_channels))
        
        self.convs.append(GATConv(hidden_channels, out_channels))
        self.bns.append(torch.nn.BatchNorm1d(norm_channels))

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        for step in range(len(self.convs)-1):
            x = self.bns[step](F.relu(self.convs[step](x, adj, mask)))

        # x = F.dropout(x, p=0.5)
        x = self.bns[step+1](F.relu(self.convs[step+1](x, adj, mask)))

        return x

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True, norm_channels=None):
        super(SAGE, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(SAGEConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(norm_channels))
        
        # self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize))
        # self.bns.append(torch.nn.BatchNorm1d(norm_channels))
        
        self.convs.append(SAGEConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(norm_channels))

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        for step in range(len(self.convs)-1):
            x = self.bns[step](F.relu(self.convs[step](x, adj, mask)))

        # x = F.dropout(x, p=0.5)
        x = self.bns[step+1](F.relu(self.convs[step+1](x, adj, mask)))

        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                norm_channels=None):
        super(GIN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(GINConv(nn.Linear(in_channels, hidden_channels)))
        self.bns.append(torch.nn.BatchNorm1d(norm_channels))
        
        # self.convs.append(GINConv(nn.Linear(hidden_channels, hidden_channels)))
        # self.bns.append(torch.nn.BatchNorm1d(norm_channels))
        
        self.convs.append(GINConv(nn.Linear(hidden_channels, out_channels)))
        self.bns.append(torch.nn.BatchNorm1d(norm_channels))

    def reset_parameters(self):
        for layer in self.convs:
            layer.reset_parameters()


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        for step in range(len(self.convs)-1):
            x = self.bns[step](F.relu(self.convs[step](x, adj, mask)))

        # x = F.dropout(x, p=0.5)
        x = self.bns[step+1](F.relu(self.convs[step+1](x, adj, mask)))

        return x

class Gen_GNN(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 normalize=False, args=None):
        super(Gen_GNN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.adj_adjust = torch.nn.ModuleList()

        self.args = args

        self.adj_adjust.append(Adj_adjust(in_channels, in_channels, module=self.args.dcrp_module))
        self.adj_adjust.append(Adj_adjust(in_channels, in_channels, module=self.args.dcrp_module))
        
        self.convs.append(GNN(in_channels, 64, in_channels, norm_channels=in_channels))
        self.convs.append(GNN(in_channels, 64, 128, norm_channels=in_channels))

        self.lin1 = nn.Linear(128, 1)
        self.lin2 = nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        for layers in self.adj_adjust:
            layers.reset_parameters()
        for layers in self.convs:
            layers.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        dcrp_flag = 1
        loss_info = 0
        for step in range(len(self.convs)):
            if self.args.dcrp and dcrp_flag:
              x_h, adj, info_loss = self.adj_adjust[step](x, adj, tau=self.args.tau, threshold=self.args.threshold)
              loss_info += info_loss
              if not self.args.multi:
                  dcrp_flag = 0
            
            x = F.relu(self.convs[step](x, adj, mask))
        x = F.dropout(x, p=0.5)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x.transpose(1, 2)))
        
        return x, loss_info

class Gen_GAT(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 normalize=False, args=None):
        super(Gen_GAT, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.adj_adjust = torch.nn.ModuleList()

        self.args = args

        self.adj_adjust.append(Adj_adjust(in_channels, in_channels, module=self.args.dcrp_module))
        self.adj_adjust.append(Adj_adjust(in_channels, in_channels, module=self.args.dcrp_module))
        
        self.convs.append(GAT(in_channels, 64, in_channels, norm_channels=in_channels))
        self.convs.append(GAT(in_channels, 64, 128, norm_channels=in_channels))

        self.lin1 = nn.Linear(128, 1)
        self.lin2 = nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        for layers in self.adj_adjust:
            layers.reset_parameters()
        for layers in self.convs:
            layers.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        dcrp_flag = 1
        loss_info = 0
        for step in range(len(self.convs)):
            if self.args.dcrp and dcrp_flag:
              x_h, adj, info_loss = self.adj_adjust[step](x, adj, tau=self.args.tau, threshold=self.args.threshold)
              loss_info += info_loss
              if not self.args.multi:
                  dcrp_flag = 0
            
            x = F.relu(self.convs[step](x, adj, mask))
        x = F.dropout(x, p=0.5)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x.transpose(1, 2)))
        
        return x, loss_info

class Gen_SAGE(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 normalize=False, args=None):
        super(Gen_SAGE, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.adj_adjust = torch.nn.ModuleList()

        self.args = args

        self.adj_adjust.append(Adj_adjust(in_channels, in_channels, module=self.args.dcrp_module))
        self.adj_adjust.append(Adj_adjust(in_channels, in_channels, module=self.args.dcrp_module))
        
        self.convs.append(SAGE(in_channels, 64, in_channels, norm_channels=in_channels))
        self.convs.append(SAGE(in_channels, 64, 128, norm_channels=in_channels))

        self.lin1 = nn.Linear(128, 1)
        self.lin2 = nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        for layers in self.adj_adjust:
            layers.reset_parameters()
        for layers in self.convs:
            layers.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        dcrp_flag = 1
        loss_info = 0
        for step in range(len(self.convs)):
            if self.args.dcrp and dcrp_flag:
              x_h, adj, info_loss = self.adj_adjust[step](x, adj, tau=self.args.tau, threshold=self.args.threshold)
              loss_info += info_loss
              if not self.args.multi:
                  dcrp_flag = 0
            
            x = F.relu(self.convs[step](x, adj, mask))
        x = F.dropout(x, p=0.5)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x.transpose(1, 2)))
        
        return x, loss_info

class Gen_GIN(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 normalize=False, args=None):
        super(Gen_GIN, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.adj_adjust = torch.nn.ModuleList()

        self.args = args

        self.adj_adjust.append(Adj_adjust(in_channels, in_channels, module=self.args.dcrp_module))
        self.adj_adjust.append(Adj_adjust(in_channels, in_channels, module=self.args.dcrp_module))
        
        self.convs.append(GIN(in_channels, 64, in_channels, norm_channels=in_channels))
        self.convs.append(GIN(in_channels, 64, 128, norm_channels=in_channels))

        self.lin1 = nn.Linear(128, 1)
        self.lin2 = nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        for layers in self.adj_adjust:
            layers.reset_parameters()
        for layers in self.convs:
            layers.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        dcrp_flag = 1
        loss_info = 0
        for step in range(len(self.convs)):
            if self.args.dcrp and dcrp_flag:
              x_h, adj, info_loss = self.adj_adjust[step](x, adj, tau=self.args.tau, threshold=self.args.threshold)
              loss_info += info_loss
              if not self.args.multi:
                  dcrp_flag = 0
            
            x = F.relu(self.convs[step](x, adj, mask))
        x = F.dropout(x, p=0.5)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x.transpose(1, 2)))
        
        return x, loss_info

class DiffPool(torch.nn.Module):
    def __init__(self, args):
        super(DiffPool, self).__init__()

        self.args = args

        # self.level_list = [args.brain_atlas, args.level1, args.level2]
        self.level_list = [args.brain_atlas, int(np.ceil(args.brain_atlas/4)), int(np.ceil(args.brain_atlas/16))]
        self.gnn_pool = nn.ModuleList()
        self.gnn_embed = nn.ModuleList()
        self.adj_adjust = nn.ModuleList()

        for index in range(len(self.level_list)-1):
            self.gnn_pool.append(GNN(self.level_list[index], 64, self.level_list[index+1], norm_channels=self.level_list[index]))
            self.gnn_embed.append(GNN(self.level_list[index], 64, self.level_list[index+1], norm_channels=self.level_list[index]))
            self.adj_adjust.append(Adj_adjust(self.level_list[index], self.level_list[index], module=self.args.dcrp_module))

        self.lin1 = nn.Linear(self.level_list[index+1], 1)
        self.lin2 = nn.Linear(self.level_list[-1], args.num_classes)

    def reset_parameters(self):
        for convs in self.gnn_pool:
            convs.reset_parameters()
        for convs in self.gnn_embed:
            convs.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    
    def forward(self, x, adj, mask=None):

        loss_l = 0
        loss_e = 0
        loss_info = 0
        dcrp_flag = 1
        logits_list = []

        for index in range(len(self.gnn_pool)):

            if self.args.dcrp and dcrp_flag:
                x_h, adj, info_loss = self.adj_adjust[index](x, adj, tau=self.args.tau, threshold=self.args.threshold)
                loss_info += info_loss
                if not self.args.multi:
                    dcrp_flag = 0
            s = self.gnn_pool[index](x, adj)
            x = self.gnn_embed[index](x, adj)
            x, adj, l_loss, e_loss = dense_diff_pool(x, adj, s)
            loss_l += l_loss
            loss_e += e_loss

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x.transpose(1, 2)))
       
        return x, loss_l+loss_e+loss_info

class TopK(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 normalize=False, args=None):
        super(TopK, self).__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.topk = torch.nn.ModuleList()
        self.adj_adjust = torch.nn.ModuleList()

        self.args = args

        self.adj_adjust.append(Adj_adjust(in_channels, in_channels, module=self.args.dcrp_module))
        self.adj_adjust.append(Adj_adjust(in_channels, int(np.ceil(in_channels/4)), module=self.args.dcrp_module))
        
        self.convs.append(GNN(in_channels, 64, in_channels, norm_channels=in_channels))
        self.topk.append(TopKPoolWeighted(in_channels, ratio=0.25))
        self.convs.append(GNN(in_channels, 64, 128, norm_channels=int(np.ceil(in_channels*0.25))))
        self.topk.append(TopKPoolWeighted(128, ratio=0.25))

        self.lin1 = nn.Linear(128, 1)
        self.lin2 = nn.Linear(int(np.ceil(np.ceil(in_channels/4)/4)), out_channels)
    
    def reset_parameters(self):
        for layers in self.adj_adjust:
            layers.reset_parameters()
        for layers in self.convs:
            layers.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()
        
        dcrp_flag = 1
        loss_info = 0
        for step in range(len(self.convs)):
            if self.args.dcrp and dcrp_flag:
              x_h, adj, info_loss = self.adj_adjust[step](x, adj, tau=self.args.tau, threshold=self.args.threshold)
              loss_info += info_loss
              if not self.args.multi:
                  dcrp_flag = 0
            
            x = F.relu(self.convs[step](x, adj, mask))
            x, adj = self.topk[step](x, adj)
            # print(step)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x.transpose(1, 2)))
        
        return x, loss_info


class TopKPoolWeighted(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super(TopKPoolWeighted, self).__init__()
        self.pool = TopKPooling(in_channels, ratio=ratio)

    def forward(self, x, adj):
        # Convert weighted adjacency matrix to edge index and edge weights
        # edge_index, edge_weight = dense_to_sparse(adj)

        data_list = []

        original_size = x.size()

        for index in range(x.size(0)):
            x_tmp = x[index].squeeze()
            adj_tmp = adj[index].squeeze()
            adj_tmp = adj_tmp + adj_tmp.t()
            # adj_tmp[adj_tmp < 1] = 0
            adj_tmp[adj_tmp>1] = 1
            edge_index_tmp, edge_weight_tmp = dense_to_sparse(adj_tmp)
            edge_index, edge_weight = add_self_loops(edge_index_tmp, edge_weight_tmp, num_nodes=x_tmp.size(0))
            data_tmp = Data(x=x_tmp, edge_index=edge_index, edge_attr=edge_weight).cuda()
            data_list.append(data_tmp)

        # data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight).cuda()

        # Create a batch (assuming you have multiple Data objects)
        # For demonstration, we just repeat the same data
        batch = Batch.from_data_list(data_list).cuda()

        # x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # Apply TopK-Pooling
        x, edge_index, edge_attr, batch, _, _ = self.pool(batch.x, batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)

        # Convert the pooled edge index back to adjacency matrices
        adj_pooled_list = []
        for i in range(batch.max().item() + 1):
            node_mask = batch == i
            num_nodes_i = node_mask.sum()
            edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
            edge_index_i = edge_index[:, edge_mask]
            adj_pooled_i = to_dense_adj(edge_index_i, max_num_nodes=num_nodes_i).squeeze(0)
            adj_pooled_list.append(adj_pooled_i)
        adj_pooled = torch.stack(adj_pooled_list)

        return x.view(original_size[0], -1, original_size[-1]).cuda(), adj_pooled.squeeze(0).cuda()


class Adj_adjust(nn.Module):
    def __init__(self, origin_level, level, module='GAT'):
        super(Adj_adjust, self).__init__()

        if module == 'GAT':
            self.embed = GATConv(origin_level, level)
        elif module == 'GCN':
            self.embed = GCNConv(origin_level, level)
        elif module == 'GIN':
            self.embed = GINConv(nn.Linear(origin_level, level))
        elif module == 'SAGE':
            self.embed = SAGEConv(origin_level, level)
        # self.gat3 = GATConv(level, origin_level)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        self.embed.reset_parameters()

    def reparameterize(self, p_i, tau=0.01, num_sample=100):
        p_i_ = p_i.view(p_i.size(0), 1, -1)
        p_i_ = p_i_.expand(p_i_.size(0), num_sample, p_i_.size(-1))
        C_sample = torch.rand(p_i_.size(), device='cuda')
        V = C_sample.mean(dim=1)
        relax_tanh = 1/(1+torch.exp(-1/tau*(V.view(p_i.size())+p_i-1)))
        # s = 1-relax_sig
        return relax_tanh

    def forward(self, x, adj, tau, threshold):
        # x = self.relu(self.gat1(x, adj))
        # print(adj)
        x = self.embed(x, adj)
        # x_prob = 1-self.softmax(x-threshold)
        x_prob = self.sigmoid(x)
        x_sample = self.reparameterize(x_prob, tau=tau)

        adj = torch.mul(adj, x_sample.view(adj.size()))

        info_loss = torch.distributions.kl.kl_divergence(torch.distributions.MultivariateNormal(x_prob,torch.eye(x_prob.size(-1)).to('cuda')),\
                     torch.distributions.MultivariateNormal(torch.ones_like(x_prob).to('cuda')*threshold,torch.eye(x_prob.size(-1)).to('cuda')))
        # print(info_loss)
        return x, adj, info_loss.sum()


def kl_divergence(p, q):
    return torch.sum(p * (torch.log(p) - torch.log(q)))