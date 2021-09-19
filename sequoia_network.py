import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GATConv, NNConv  # noqa
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.data import DataLoader

class Sequoia(torch.nn.Module):
    # Message passing neural network composed of 2 sequential blocks. Each block uses a continuous kernel-based convolutional operator with 4 linear layers with ReLU activations.
    def __init__(self, num_node_features, num_classes, num_edge_features):
        super(Sequoia, self).__init__()

        ##############
        # Parameters controlling the size of the linear operators and convolutional filters 
        param0, param1, param2, param3, param4 = 8, 64, 64, 64, 64  # First block
        param5, param6, param7, param8, param9 = 64, 25, 25, 25, 64         # Second block
        ##############

        ##############
        # First block
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, param1), torch.nn.ReLU(), torch.nn.Linear(param1, param3), torch.nn.ReLU(), torch.nn.Linear(param3, param4), torch.nn.ReLU(), torch.nn.Linear(param4, num_node_features*param2))
        self.conv1 = NNConv(num_node_features, param5, nn1, aggr='mean')
        ##############

        ##############
        # Second block
        nn2 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, param6), torch.nn.ReLU(), torch.nn.Linear(param6, param7), torch.nn.ReLU(), torch.nn.Linear(param7, param8), torch.nn.ReLU(), torch.nn.Linear(param8, param5*param9))
        self.conv2 = NNConv(param5, param9, nn2, aggr='mean')
        ##############

        ##############
        # Intermerdiate operations
        self.fc1 = torch.nn.Linear(param2, param3)
        self.fc2 = torch.nn.Linear(param3, num_classes)
        ##############

    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)

