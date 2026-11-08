import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        # Adjust padding for 'same' convolution
        padding = (0, (kernel_size - 1) // 2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=padding)

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)  # (B, C_in, N, T)
        temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)  # (B, N, T, C_out)
        return out

class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix (num_nodes x num_nodes).
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        t = self.temporal1(X)  # (B, N, T1, C_out)
        # lfs: apply graph convolution over nodes
        # t.permute(1,0,2,3) -> (N, B, T1, C_out)
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # lfs: (B, T1, C_out) for each node -> after matmul with Theta1 we get spatial_channels
        t2 = F.relu(torch.matmul(lfs, self.Theta1))  # (B, T1, spatial_channels)
        t3 = self.temporal2(t2)  # (B, N, T2, out_channels)
        return self.batch_norm(t3)


class STGCN(nn.Module):
    """
    Spatio-temporal graph convolutional network as described in
    https://arxiv.org/abs/1709.04875v3 by Yu et al.
    Input should have shape (batch_size, num_nodes, num_input_time_steps,
    num_features).
    """

    def __init__(self, nums_node, nums_step_in, nums_step_out, nums_feature_in, nums_feature_out, isolated_nodes=None):
        """
        :param num_nodes: Number of nodes in the graph.
        :param num_features: Number of features at each node in each time step (input features).
        :param num_timesteps_input: Number of past time steps fed into the network.
        :param num_timesteps_output: Desired number of future time steps output by the network.
        :param num_features_output: Number of features to predict per output timestep.
        """
        super(STGCN, self).__init__()
        self.isolated_nodes = sorted(set(int(i) for i in (isolated_nodes or [])))
        self.block1 = STGCNBlock(in_channels=nums_feature_in, out_channels=64,
                                 spatial_channels=16, num_nodes=nums_node)
        self.block2 = STGCNBlock(in_channels=64, out_channels=64,
                                 spatial_channels=16, num_nodes=nums_node)
        self.last_temporal = TimeBlock(in_channels=64, out_channels=64)

        # number of values per node after temporal blocks (flatten length)
        # Keep the original reduction formula but you can adjust if kernel sizes differ.
        

        # Map flattened per-node features to (T_out * features_out)
        self.num_timesteps_output = nums_step_out
        self.num_features_output = nums_feature_out
        self.fully = nn.Linear(nums_step_in * 64, nums_step_out * nums_feature_out)

        self.config = {
            "nums_node": nums_node,
            "nums_step_in": nums_step_in,
            "nums_step_out": nums_step_out,
            "nums_feature_in": nums_feature_in,
            "nums_feature_out": nums_feature_out,
            "isolated_nodes": self.isolated_nodes,
        }
        
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
        :param A_hat: Normalized adjacency matrix (num_nodes x num_nodes)
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features=in_channels).
        :return: Tensor of shape (batch_size, num_nodes, num_timesteps_output, num_features_output)
        """
        A_hat = self._cut(A_hat)
        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1, A_hat)
        out3 = self.last_temporal(out2)  # (B, N, T_reduced, 64)

        # flatten temporal+channel dims per node
        B, N, T_r, C = out3.shape
        assert C == 64, "unexpected channel size after last temporal block"

        flat = out3.reshape((B, N, -1))  # (B, N, T_r * 64)
        out4 = self.fully(flat)  # (B, N, T_out * features_out)

        # reshape to (B, N, T_out, features_out)
        out4 = out4.view(B, N, self.num_timesteps_output, self.num_features_output)
        return out4

    def forward_unlearning(self, X):
        """Forward pass for unlearning flow (uses stored adjacency matrix).
           Accept different input shapes:
             - (B, N, T)
             - (N, T)
             - (B, N, T, F)
        """
        # Handle different input shapes
        if X.dim() == 3:  # (B, N, T) -> add feature dimension
            X = X.unsqueeze(-1)
        elif X.dim() == 2:  # (N, T) -> add batch and feature dimensions
            X = X.unsqueeze(0).unsqueeze(-1)

        # Use stored adjacency matrix
        if not hasattr(self, '_stored_A_hat'):
            raise ValueError("Adjacency matrix not set. Call set_adjacency_matrix() first.")

        return self.forward(self._stored_A_hat, X)

    def set_adjacency_matrix(self, A_hat):
        """Store adjacency matrix for unlearning"""
        self._stored_A_hat = A_hat