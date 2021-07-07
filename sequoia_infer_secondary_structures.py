import os.path as osp
import argparse
import json, random
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GATConv, NNConv  # noqa
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import pdb, random, time
from sklearn.metrics import f1_score
import sequoia_network

#class Net2(torch.nn.Module):
#    def __init__(self, num_features, num_classes, num_edge_features):
#        super(Net2, self).__init__()
#        #param1 = 32
#        #param2 = 128
#        param0 = 8
#        param1 = 64 
#        param2 = 64
#        param3 = 64
#        param4 = 64
#        param5 = 64
#
#        nn1 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, param1), torch.nn.ReLU(), torch.nn.Linear(param1, param3), torch.nn.ReLU(), torch.nn.Linear(param3, param4), torch.nn.ReLU(), torch.nn.Linear(param4, num_node_features*param2))
#
#
#        self.conv1 = NNConv(num_node_features, param2, nn1, aggr='mean')
#
#        nn2 = torch.nn.Sequential(torch.nn.Linear(num_edge_features, 25), torch.nn.ReLU(), torch.nn.Linear(25, 25), torch.nn.ReLU(), torch.nn.Linear(25, 25), torch.nn.ReLU(), torch.nn.Linear(25, param2*param2))
#
#        self.conv2 = NNConv(param2, param2, nn2, aggr='mean')
#        
#
#        self.fc1 = torch.nn.Linear(param2, param3)
#        self.fc2 = torch.nn.Linear(param3, num_classes)
#
#    def forward(self, data):
#        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
#        x = F.elu(self.conv2(x, data.edge_index, data.edge_attr))
#        #x = F.elu(self.conv3(x, data.edge_index, data.edge_attr))
#        x = F.elu(self.fc1(x))
#        x = F.dropout(x, training=self.training)
#        return F.log_softmax(self.fc2(x), dim=1)




@torch.no_grad()
def testNNConv(other_test=False, data_test=[]):
    model.eval()
    PREDS = []
    if not other_test:
        logits, scores = model(data), []
        for mask_key, mask in data('test_mask'):
        #for mask_key, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            score = f1_score(pred.tolist(), data.y[mask].tolist(), average="micro")
            scores.append(score)
            if mask_key == "test_mask":
                PREDS = pred.tolist()

    else:
        logits_train, _ = model(data), []
        logits_test, scores = model(data_test), []
        for mask_key, mask in data('train_mask', 'val_mask'): # , 'test_mask'):
            #if "test" not in mask:
            if 1:
                pred = logits_train[mask].max(1)[1]
                score = f1_score(pred.tolist(), data.y[mask].tolist(), average="micro")
                scores.append(score)
        
        for mask_key, mask in data_test("test_mask"): 
            pred = logits_test[mask].max(1)[1]
            score = f1_score(pred.tolist(), data_test.y[mask].tolist(), average="micro")
            scores.append(score)
            PREDS = pred.tolist()

    return [scores, PREDS]

def noise_project(X, epsilon):
    sign_X = np.sign(X)
    first_condition = (np.array(X+epsilon) <= 1 )*(np.array(X+epsilon) >= -1) 
    second_condition = np.array(X+epsilon) > 1
    third_condition = np.array(X+epsilon) < -1

    first_case = (X+epsilon)*first_condition
    second_case = (X+epsilon-2)*second_condition
    third_case = (2+(X+epsilon))*third_condition

    return first_case + second_case + third_case


def ablateEdges(As_I, NXs_I, proportion_ablation):
    # Ablation study: remove some residues from Adjacency matrix and corresponding edge features
    if proportion_ablation > 0:
        NXs_F_ablated = []
        As_F_ablated = []
        for i_protein, A in enumerate(As_I):
            A_ablated = A.copy()
            range_indices = max(A[0])
            n_samples = int((proportion_ablation/100)*range_indices)
            indices_to_remove = random.sample(range(range_indices), n_samples)
            indices_to_keep = [i for i in range(range_indices) if i not in indices_to_remove]
            edges_indices_to_keep = [i_edge for i_edge, edge in enumerate(zip(A[0], A[1])) if not (edge[1] in indices_to_remove or edge[0] in indices_to_remove)]
                                                                                                                                                               
            A_ablated = [[A[0][eitk] for eitk in edges_indices_to_keep], [A[1][eitk] for eitk in edges_indices_to_keep]]
            NX_ablated = [NXs_I[i_protein][eitk] for eitk in edges_indices_to_keep]
                                                                                                                                                               
            As_F_ablated.append(A_ablated)
            NXs_F_ablated.append(NX_ablated)
        return As_F_ablated, NXs_F_ablated

    else:
        return As_I, NXs_I





if __name__ == "__main__":
    import numpy as np
    import edge_multi_load_multiBio, calpha_edge_multi_load_multiBio, new_edge_multi_load_multiBio, final_edge_multi_load_multiBio
    import pdb
    import sys
    import pickle
    
    filename = sys.argv[1]
    classification_type = sys.argv[2]
    model_path = sys.argv[3]
    calpha_mode = int(sys.argv[4])
    dssp_mode = int(sys.argv[5])
    #model_path = "/data/PDB/cullpdb/ablation_gnn_trained_" + classification_type  + calpha_str + "cullpdb_nbepochs_" + str(nb_epochs) + ".tch"
    write_directory = sys.argv[6]

    try:
        name_to_pattern_filename = sys.argv[7]
        name_to_pattern = json.load(open(name_to_pattern_filename, "r"))
    except:
        print("Warning: no conformation table provided, could be required for protein with several conformations")
        name_to_pattern = {} 

    As_0, Xs, Ys_0, NXs_0_init, _ = final_edge_multi_load_multiBio.loadGraphFromFile(filename, name_to_pattern, classification_type=classification_type, distance_based=True, shuffle=False, nmr_conformations=False, calpha_mode=calpha_mode, dssp_mode=dssp_mode)

    As_0 = [As_0]
    Ys_0 = [Ys_0]
    Xs = [Xs]
    NXs_0_init = [NXs_0_init]

    NXs_0 = []
    import time
    start = time.time()

    if classification_type == "helices":
        map_st = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5:1, 6: 1, 7: 1}
        num_classes = 2
 
    if classification_type == "all":
        map_st = {i:i for i in range(8)}
        num_classes = 8

    if classification_type == "sheets":
        map_st = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 0, 7: 0}
        num_classes = 2

    if classification_type == "helices_sheets":
        map_st = {0: 0, 1: 0, 2: 0, 3: 2, 4: 1, 5: 2, 6: 2, 7: 2}
        num_classes = 3

    ##############################################################################################################
    ############### Data augmentation: aggregate neighborhood  edge features into node features ##################
    As = As_0
    NXs = NXs_0_init
    Ys = Ys_0

    amplitude_noise = 0
    if 0:
        NXs_0 = []
        import time
        start = time.time()
        for protein_NX in NXs:
            noise = amplitude_noise*np.random.randn(len(protein_NX),2)
            protein_NX_array = np.array(protein_NX)
            noisy_protein_NX_array = noise_project(protein_NX_array, noise)
            NXs_0.append(noisy_protein_NX_array.tolist())
        NXs = NXs_0.copy()

    end = time.time()

    #
    #X_phi_psi_0 = []
    #for i_PA, PA in enumerate(As):
    #    X_phi_psi_0_D = {}
    #    for i_edge, source in enumerate(PA[0]):
    #        if i_edge % 2 == 0:
    #            feature_source = NXs[i_PA][i_edge]
    #            current_feature_source = X_phi_psi_0_D.get(source, []) 
    #            current_feature_source += feature_source 
    #            X_phi_psi_0_D[source] = current_feature_source
    #    
    #    for source_tmp in X_phi_psi_0_D.keys():
    #        if len(X_phi_psi_0_D[source_tmp]) < 4: # default dimension for nb_neighbors = 2
    #            len_miss = 4 - len(X_phi_psi_0_D[source_tmp])
    #            value_tmp = X_phi_psi_0_D.get(source_tmp, [])
    #            X_phi_psi_0_D[source_tmp] = value_tmp + [0. for i_tmp in range(len_miss)]

    #    X_phi_psi_P = [[0., 0., 0., 0.] for i in range(len(Xs[i_PA][:-1]))]  # default dimension for nb_neighbors = 2
    #    for key in X_phi_psi_0_D:
    #        X_phi_psi_P[key] = X_phi_psi_0_D[key]
    #    X_phi_psi_0.append(X_phi_psi_P)

    #Xs = [np.concatenate([X_[:-1], X_phi_psi_0[i]], axis=1) for i, X_ in enumerate(Xs)]

    #if len(Ys_0[0]) > 0:
    #    ground_truth_provided = True
    #    Ys = [np.array([map_st[yy] for yy in y.tolist()]) for y in Ys_0]
    #    Ys = [Y_[:-1] for Y_ in Ys]
    #    Y_values = [Y.tolist() for Y in Ys]
    #    Y_values = sum(Y_values, [])
    #    Y_values = sorted(list(set(Y_values)))
    #    Y_tmp = []
    #    Y_map = {}
    #    for i, vy in enumerate(Y_values):
    #        Y_map[vy] = i
    #    Ys = [np.array([Y_map[y] for y in Y]) for Y in Ys]
    #else:
    #    print("No ground truth provided")
    #    ground_truth_provided = False
    #    Ys = [np.array([0] + [1 for i in range(len(Xs[0]) - 1)])]
    if classification_type == "helices":
        nb_labels = 2
    if classification_type == "helices_sheets":
        nb_labels = 3

    As, Xs, Ys, NXs, ground_truth_provided = final_edge_multi_load_multiBio.dataAugmentation(As, Xs, Ys, NXs, nb_labels, map_st)

    ##############################################################################################################
    ##############################################################################################################

    data_batches, indices_protein = edge_multi_load_multiBio.graphListToData(As, Xs, Ys, NXs, test=False)
    data = data_batches

    n_mask = len(data.test_mask)
    for i in range(n_mask):
        data.test_mask[i] = True
        data.val_mask[i] = False
        data.train_mask[i] = False

    num_node_features = len(Xs[0][0])
    num_edge_features = len(NXs[0][0])
    ##############################################################################
    ##############################################################################

    from collections import Counter
    np.set_printoptions(threshold=sys.maxsize)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model, data = Net2(num_node_features, num_classes, num_edge_features).to(device), data.to(device)
    model, data = sequoia_network.Sequoia(num_node_features, num_classes, num_edge_features).to(device), data.to(device)



    PREDS = []
    FINAL_SCORES = []
    best_val_acc = test_acc = 0
    calpha_str = "_"
    if calpha_mode:
        calpha_str = "_calpha_"

    #print(num_classes)
    
    pdb.set_trace()
    if osp.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        test_acc, preds = testNNConv(other_test=False) #, data_test=data_test)
        PREDS.append(preds)
        print(preds)
        print(Ys[0])
        assert(len(preds) == len(Ys[0]))
        filename_split_slash = filename.split("/")
        if len(filename_split_slash) > 1:
            filename_w = filename_split_slash[-1]
            filename_w = filename_w.split(".")[0]
        else:
            filename_w = filename.split(".")[0]

        with open(write_directory + filename_w + "_" + classification_type + ".preds", "w") as f:      
            f.write(" ".join([str(pred) for pred in preds]))  

        if ground_truth_provided: 
            common_labels = 100*len([1 for i in range(len(preds)) if preds[i] == Ys[0][i]])/float(len(preds))
            print("Common labels: " + str(common_labels) + " %")



    
