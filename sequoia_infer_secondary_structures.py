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
    output_filename = sys.argv[6]

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

    data_batches, indices_protein = sequoia_dataload_multibio.graphListToData(As, Xs, Ys, NXs, test=False)
    data = data_batches

    n_mask = len(data.test_mask)
    for i in range(n_mask):
        data.test_mask[i] = True
        data.val_mask[i] = False
        data.train_mask[i] = False

    num_node_features = len(Xs[0][0])
    num_edge_features = len(NXs[0][0])
    ##############################################################################################################
    ##############################################################################################################

    from collections import Counter
    np.set_printoptions(threshold=sys.maxsize)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = sequoia_network.Sequoia(num_node_features, num_classes, num_edge_features).to(device), data.to(device)


    PREDS = []
    FINAL_SCORES = []
    best_val_acc = test_acc = 0
    calpha_str = "_"
    if calpha_mode:
        calpha_str = "_calpha_"

    if osp.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        test_acc, preds = testNNConv(other_test=False) #, data_test=data_test)
        PREDS.append(preds)
        print(preds)
        assert(len(preds) == len(Ys[0]))
        filename_split_slash = filename.split("/")
        if len(filename_split_slash) > 1:
            filename_w = filename_split_slash[-1]
            filename_w = filename_w.split(".")[0]
        else:
            filename_w = filename.split(".")[0]

        with open(output_filename, "w") as f:      
            f.write(" ".join([str(pred) for pred in preds]))  

    
