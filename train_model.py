import json, random
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GATConv, NNConv  # noqa
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import pdb, random, time
from sklearn.metrics import f1_score

class Sequoia(torch.nn.Module):
    # Message passing neural network composed of 2 sequential blocks. Each block use a continuous kernel-based convolutional operator with 4 linear layers with ReLU activations.
    def __init__(self, num_features, num_classes, num_edge_features):
        super(Net2, self).__init__()
        param0, param1, param2, param3, param4, param5 = 8, 64, 64, 64, 64, 64 # parameters controling linear transformations size 

        ##############
        # First block
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, param1), torch.nn.ReLU(), torch.nn.Linear(param1, param3), torch.nn.ReLU(), torch.nn.Linear(param3, param4), torch.nn.ReLU(), torch.nn.Linear(param4, num_node_features*param2))
        self.conv1 = NNConv(num_node_features, param2, nn1, aggr='mean')
        ##############

        ##############
        # Second block
        nn2 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, 25), torch.nn.ReLU(), torch.nn.Linear(25, 25), torch.nn.ReLU(), torch.nn.Linear(25, 25), torch.nn.ReLU(), torch.nn.Linear(25, param2*param2))
        self.conv2 = NNConv(param2, param2, nn2, aggr='mean')
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



def trainNNConv(model):
    model.train()
    optimizer.zero_grad()
    out_model = model(data)
    loss = F.nll_loss(out_model[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def testNNConv(other_test=False, data_test=[]):
    model.eval()
    PREDS = []
    if not other_test:
        logits, scores = model(data), []
        for mask_key, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            score = f1_score(pred.tolist(), data.y[mask].tolist(), average="micro")
            scores.append(score)

    return scores

def noise_project(X, epsilon):
    sign_X = np.sign(X)
    first_condition = (np.array(X+epsilon) <= 1 )*(np.array(X+epsilon) >= -1) 
    second_condition = np.array(X+epsilon) > 1
    third_condition = np.array(X+epsilon) < -1

    first_case = (X+epsilon)*first_condition
    second_case = (X+epsilon-2)*second_condition
    third_case = (2+(X+epsilon))*third_condition

    return first_case + second_case + third_case


if __name__ == "__main__":
    import numpy as np
    import edge_multi_load_multiBio, calpha_edge_multi_load_multiBio
    import pdb
    import sys
    import pickle
    
    filename = sys.argv[1]
    classification_type = sys.argv[2]
    noise_level = float(sys.argv[3])
    nb_neighbors = int(sys.argv[4])
    model_path_output = sys.argv[5]
    assert(noise_level <= 0.5 and noise_level >= 0)
    
    with open(filename, "rb") as f:
        As_0, Xs, Ys_0, NXs_0_init, _ = pickle.load(f)
            
    ##############################################################################
    if classification_type == "all":
        map_st = {i:i for i in range(8)}

    if classification_type == "helices":
        #map_st = {"G": 0, "H": 0, "I": 0, "T": 1, "E": 1, "B": 1, "S": 1, "-": 1 }
        #map_st = {"G": 0, "H": 1, "I": 2, "T": 3, "E": 4, "B": 5, "S": 6, "-": 7}
        map_st = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5:1, 6: 1, 7: 1}

    if classification_type == "sheets":
        map_st = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0}


    if classification_type == "helices_sheets":
        #map_st = {"G": 0, "H": 0, "I": 0, "T": 2, "E": 1, "B": 2, "S": 2, "-":  2}
        map_st = {0: 0, 1: 0, 2: 0, 3: 2, 4: 1, 5: 2, 6: 2, 7: 2}

    
    #############################################################################
    ########################## Data augmentation ###############################
    As = As_0
    NXs = NXs_0
    Ys = [np.array([map_st[yy] for yy in y.tolist()]) for y in Ys_0]
    X_phi_psi_0 = []
    for i_PA, PA in enumerate(As):
        X_phi_psi_0_D = {}
        for i_edge, source in enumerate(PA[0]):
            if i_edge % 2 == 0:
                feature_source = NXs[i_PA][i_edge]
                current_feature_source = X_phi_psi_0_D.get(source, []) 
                current_feature_source += feature_source 
                X_phi_psi_0_D[source] = current_feature_source
            
        X_phi_psi_P = [[] for key in X_phi_psi_0_D]
        for key in X_phi_psi_0_D:
            X_phi_psi_P[key] = X_phi_psi_0_D[key]
        X_phi_psi_0.append(X_phi_psi_P)

    Xs = [np.concatenate([X_[:-1], X_phi_psi_0[i]], axis=1) for i, X_ in enumerate(Xs)]
    #############################################################################

    Ys = [Y_[:-1] for Y_ in Ys]
    Y_values = [Y.tolist() for Y in Ys]
    Y_values = sum(Y_values, [])
    Y_values = sorted(list(set(Y_values)))
    Y_tmp = []
    Y_map = {}
    for i, vy in enumerate(Y_values):
        Y_map[vy] = i
    Ys = [np.array([Y_map[y] for y in Y]) for Y in Ys]
                                                                                              
    data_batches, indices_protein = edge_multi_load_multiBio.graphListToData(As, Xs, Ys, NXs)
    data = data_batches

                                                                                              
    num_node_features = len(Xs[0][0])
    num_edge_features = len(NXs[0][0])
                                                                                              
    num_classes =  len(set(Y_values))
    Yvalues = [Y.tolist() for Y in Ys]
    Yvalues=sum(Yvalues, [])


    ##############################################################################
    ##############################################################################

    from collections import Counter
    print(Counter(Yvalues))
    np.set_printoptions(threshold=sys.maxsize)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    
    model, data = Sequoia(num_node_features, num_classes, num_edge_features).to(device), data.to(device)

    optimizer = torch.optim.Adam([
        dict(params=model.parameters()),
    ], lr=0.005)  

    
    best_val_acc = test_acc = 0
    nb_epochs = 100 
    model_path = model_path_output

    for epoch in range(1, nb_epochs+1):
        trainNNConv(model)
        [train_acc, val_acc, tmp_test_acc] = testNNConv()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))

    torch.save(model.state_dict(), model_path)
