import os.path as osp
import json
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, GATConv, NNConv  # noqa
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
import pdb, random, time
from sklearn.metrics import f1_score


if __name__ == "__main__":
    import numpy as np
    import load_multiBio, multi_load_multiBio, edge_multi_load_multiBio
    import pdb
    import sys
    import pickle
    
    filedir = sys.argv[1]
    name_output = sys.argv[2]
    nb_neighbors = int(sys.argv[3])
    conformation = int(sys.argv[4])
    if conformation:
        try:
            filename_conformation = sys.argv[5]
            name_to_pattern = json.load(open(filename_conformation, "r")) 
        except:
            print("Error: no valid conformation table was provided")
            sys.exit()
    else:
        name_to_pattern = {}

    assert(len(name_output) > 0)

    As, Xs, Ys, NXs, FOSs = edge_multi_load_multiBio.loadGraphsFromDirectory(filedir, name_to_pattern, distance_threshold=3.5, nb_neighbors=nb_neighbors, shuffle=True, nmr_conformations=nmr_conformations)
    with open(name_output + ".pkl", "wb") as f:
        pickle.dump([As, Xs, Ys, NXs, FOSs], f)

