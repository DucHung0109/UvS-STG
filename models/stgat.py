import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.nn as nn

class STGAT(torch.nn.Module):
    def __init__(self, nums_node, nums_step_in, nums_step_out, nums_feature_in, nums_feature_out, n_heads=8, dropout=0.0, isolated_nodes=None):
        """
        Initialize the ST-GAT model
        :param in_channels Number of input channels
        :param out_channels Number of output channels
        :param n_nodes Number of nodes in the graph
        :param heads Number of attention heads to use in graph
        :param dropout Dropout probability on output of Graph Attention Network
        """
        super(STGAT, self).__init__()
        self.isolated_nodes = sorted(set(int(i) for i in (isolated_nodes or [])))
        self.num_timesteps_output = nums_step_out
        self.heads = n_heads
        self.dropout = dropout
        self.n_nodes = nums_node
        self.edge_index = None
        self.n_preds = 9
        self.hidden_dim = 32

        self.config = {
            "nums_node": nums_node,
            "nums_step_in": nums_step_in,
            "nums_step_out": nums_step_out,
            "nums_feature_in": nums_feature_in,
            "nums_feature_out": nums_feature_out,
            "n_heads": n_heads,
            "dropout": dropout,
            "isolated_nodes": self.isolated_nodes,
        }

        # single graph attentional layer with 8 attention heads
        self.gat = GATConv(in_channels=nums_feature_in, out_channels=self.hidden_dim,
            heads=n_heads, dropout=0, concat=False)
        
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,     # 2 layers LSTM
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_dim, nums_feature_out * nums_step_out)

    def process(self, W):
        edge_index = torch.nonzero(W, as_tuple=False).t() #(2, num_edges)
        self.edge_index = edge_index.to(device=W.device)

    def isolate_nodes(self, nodes):
        """Cut nodes from the fixed supports and the learned adjacency.
        Stored in self.config, so a saved and reloaded model stays isolated."""
        nodes = [nodes] if isinstance(nodes, int) else list(nodes)
        self.isolated_nodes = sorted(set(self.isolated_nodes) | set(int(i) for i in nodes))
        self.config["isolated_nodes"] = self.isolated_nodes

    def _cut(self, A):
        """Zero the rows and columns of the isolated nodes."""
        if not self.isolated_nodes:
            return A
        keep = torch.ones(A.shape[0], dtype=A.dtype, device=A.device)
        keep[self.isolated_nodes] = 0.0
        return A * keep[:, None] * keep[None, :]

    
    def forward(self, A_hat, X):
        """
        Forward pass of the ST-GAT model
        :param data Data to make a pass on
        :param device Device to operate on
        """
        if self.edge_index != None:
            edge_index = self.edge_index
        else:
            A_hat = self._cut(A_hat)
            self.process(A_hat)
            edge_index = self.edge_index
        
        B, N, T_in, F = X.shape  
        X_reshaped = X.permute(2, 0, 1, 3)  # (T, B, N, F)
        X_reshaped = X_reshaped.reshape(T_in, B*N, F)  # (T, B*N, F)
        outs = []
        for t in range(T_in):
            x_t = X_reshaped[t]
            h_t = self.gat(x_t, edge_index)
            h_t = h_t.reshape(B, N, -1)
            outs.append(h_t.unsqueeze(2))

        # stack timesteps → (B, N, T_in, hidden_dim)
        outs = torch.cat(outs, dim=2)

        lstm_in = outs.reshape(B * N, T_in, self.hidden_dim)
        lstm_out, _ = self.lstm(lstm_in)
        h_last = lstm_out[:, -1, :]

        # FC to predict F * T_out
        out = self.fc(h_last)
        out = out.reshape(B, N, self.num_timesteps_output, F)  # (B, N, T_out, F)
        return out