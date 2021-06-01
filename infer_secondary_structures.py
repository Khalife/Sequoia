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
from sequoia_network import Sequoia


@torch.no_grad()
def testNNConv(data):
    model.eval()
    PREDS = []
    logits, scores = model(data), []
    for mask_key, mask in data('test_mask'):
        pred = logits[mask].max(1)[1]
        score = f1_score(pred.tolist(), data.y[mask].tolist(), average="micro")
        scores.append(score)
        if mask_key == "test_mask":
            PREDS = pred.tolist()

    return [scores, PREDS]

if __name__ == "__main__":
    import numpy as np
    import sequoia_dataload_multibio
    import pdb
    import sys
    import pickle
    
    filename = sys.argv[1]
    classification_type = sys.argv[2]
    model_path = sys.argv[3]
    calpha_mode = int(sys.argv[4])
    dssp_mode = int(sys.argv[5])

    try:
        name_to_pattern_filename = sys.argv[6]
        name_to_pattern = json.load(open(name_to_pattern_filename, "r"))
    except:
        print("Warning: no conformation table provided, could be required for protein with several conformations")
        name_to_pattern = {} 


    ##############################################################################################################
    ##############################################################################################################
    As_0, Xs, Ys_0, NXs_0_init, _ = sequoia_dataload_multibio.loadGraphFromFile(filename, name_to_pattern, classification_type=classification_type, calpha_mode=calpha_mode, dssp_mode=dssp_mode)

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

    if classification_type == "helices":
        nb_labels = 2
    As, Xs, Ys, NXs, ground_truth_provided = sequoia_dataload_multibio.dataAugmentation(As, Xs, Ys, NXs, nb_labels, map_st)

    ##############################################################################################################
    ##############################################################################################################
    data_batches, indices_protein = sequoia_dataload_multibio.graphListToData(As, Xs, Ys, NXs, test=False)
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
    model, data = Sequoia(num_node_features, num_classes, num_edge_features).to(device), data.to(device)

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
        print(Ys[0])
        assert(len(preds) == len(Ys[0]))
        with open(filename.split(".")[0] + ".preds", "w") as f:      
            f.write(" ".join([str(pred) for pred in preds]))  

        if ground_truth_provided: 
            common_labels = 100*len([1 for i in range(len(preds)) if preds[i] == Ys[0][i]])/float(len(preds))
            print("Common labels: " + str(common_labels) + " %")



    
