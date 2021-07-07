import json, random
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GATConv, NNConv  # noqa
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch_geometric.data import DataLoader
import pdb, random, time
from sklearn.metrics import f1_score

class Sequoia(torch.nn.Module):
    # Message passing neural network composed of 2 sequential blocks. Each block uses a continuous kernel-based convolutional operator with 4 linear layers with ReLU activations.
    def __init__(self, num_node_features, num_classes, num_edge_features):
        super(Sequoia, self).__init__()

        ##############
        # Parameters controlling the size of the linear operators and convolutional filters 
        param0, param1, param2, param3, param4 = 8, 64, 64, 64, 64  # First block
        param5, param6, param7, param8, param9 = 25, 25, 25, 25, 64         # Second block
        ##############

        ##############
        # First block
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, param1), torch.nn.ReLU(), torch.nn.Linear(param1, param3), torch.nn.ReLU(), torch.nn.Linear(param3, param4), torch.nn.ReLU(), torch.nn.Linear(param4, num_node_features*param5))
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


def trainNNConvBatch(model, data_loader, size_batch=2):
    optimizer.zero_grad()
    for data_batch in data_loader:
        out_model = model(data_batch)
        loss = F.nll_loss(out_model[data_batch.train_mask], data_batch.y[data_batch.train_mask])
        loss.backward()
        optimizer.step()

@torch.no_grad()
def testNNConvBatch(model, data_loader, other_test=False, data_test=[]):
    model.eval()
    PREDS = {'train_mask': [], 'val_mask': [], 'test_mask': []}
    GROUND_TRUTH_PREDS = {'train_mask': [], 'val_mask': [], 'test_mask': []}
    scores = [] 
    if not other_test:
        for data_batch in data_loader: 
            logits, _ = model(data_batch), []
            for mask_key, mask in data_batch('train_mask', 'val_mask', 'test_mask'):
                pred = logits[mask].max(1)[1]
                PREDS[mask_key] += pred.tolist()
                GROUND_TRUTH_PREDS[mask_key] += data_batch.y[mask].tolist()

        for mask_key in ['train_mask', 'val_mask', 'test_mask']:
            score = f1_score(PREDS[mask_key], GROUND_TRUTH_PREDS[mask_key], average="micro")
            scores.append(score)

    return scores, [PREDS["train_mask"], PREDS["val_mask"], PREDS["test_mask"]]


if __name__ == "__main__":
    import numpy as np
    import sequoia_multi_data_loader
    import pdb, sys, pickle
    
    filename = sys.argv[1]
    classification_type = sys.argv[2]
    nb_neighbors = int(sys.argv[3])
    model_path_output = sys.argv[4]
    
    with open(filename, "rb") as f:
        As_0, Xs, Ys_0, NXs_0_init, _ = pickle.load(f)

    ##############################################################################
    #map_st = {"G": 0, "H": 1, "I": 2, "T": 3, "E": 4, "B": 5, "S": 6, "-": 7}
    if classification_type == "all":
        map_st = {i:i for i in range(8)}
        nb_labels = 8

    if classification_type == "helices":
        map_st = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5:1, 6: 1, 7: 1}
        nb_labels = 2

    if classification_type == "sheets":
        map_st = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0}
        nb_labels = 2

    if classification_type == "helices_sheets":
        map_st = {0: 0, 1: 0, 2: 0, 3: 2, 4: 1, 5: 2, 6: 2, 7: 2}
        nb_labels = 3

    
    #############################################################################
    ########################## Data augmentation ###############################
    As, Xs, Ys, NXs, ground_truth_provided = sequoia_multi_data_loader.dataAugmentation(As_0, Xs, Ys_0, NXs_0_init, nb_labels, map_st)
    data_batches, indices_protein = sequoia_multi_data_loader.graphListToData(As, Xs, Ys, NXs)
    data = data_batches
                                                                                              
    num_node_features = len(Xs[0][0])
    num_edge_features = len(NXs[0][0])
                                                                                              
    #num_classes =  len(set(Y_values))
    num_classes = nb_labels
    Yvalues = [Y.tolist() for Y in Ys]
    Yvalues=sum(Yvalues, [])
    ##############################################################################
    ##############################################################################

    from collections import Counter
    print(Counter(Yvalues))
    np.set_printoptions(threshold=sys.maxsize)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model, data = Sequoia(num_node_features, num_classes, num_edge_features).to(device), data.to(device)
    data_list, _ =  sequoia_multi_data_loader.graphListToDataList(As, Xs, Ys, NXs)
    size_batch = 2
    data_loader = DataLoader([sequoia_multi_data_loader.returnDataFromDataList(x_dl[0], x_dl[1], x_dl[2], x_dl[3]).to(device) for x_dl in data_list], batch_size=size_batch)

    print(model)

    optimizer = torch.optim.Adam([
        dict(params=model.parameters()),
    ], lr=0.00005)  

    
    best_val_acc = test_acc = 0
    nb_epochs = 50 
    model_path = model_path_output

    for epoch in range(1, nb_epochs+1):
        trainNNConvBatch(model, data_loader, size_batch=size_batch)
        [train_acc, val_acc, tmp_test_acc], PREDS = testNNConvBatch(model, data_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

    torch.save(model.state_dict(), model_path)

