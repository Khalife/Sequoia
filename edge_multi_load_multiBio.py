import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.data import Batch
import random
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import *
import Bio
from Bio import PDB
from Bio.Seq import Seq
from scipy.spatial import distance as scipy_distance
import pdb, os, sys, random, signal, time, warnings


def graphToData(A, X, Y, NX, test=False):
    # A adjencency matrix
    # M nodes feature matrix
    # Y Labels
    nv =Y.shape[0]
    At = torch.tensor(A, dtype=torch.long)
    Xt = torch.tensor(X, dtype=torch.float)
    NXt = torch.tensor(NX, dtype=torch.float)
    data = Data(edge_index=At, x=Xt, edge_attr=NXt)
    data.y = torch.tensor(Y)

    if not test:
        mask_tmp = [i for i in range(nv)]
        random.shuffle(mask_tmp)
        nb_train = int(29*nv/40) # proportion of train / val / test are arbitrary
        nb_val = int(3.*nv/4)
        fraction_train, fraction_val, fraction_test = mask_tmp[:nb_train], mask_tmp[nb_train:nb_val], mask_tmp[nb_val:] 
        indices_train = [False for i in range(nv)]
        indices_val = [False for i in range(nv)]
        indices_test = [False for i in range(nv)]
        for ft1 in fraction_train:                
            indices_train[ft1] = True                
                                                     
        for fv in fraction_val:                      
            indices_val[fv] = True                  
                                                             
        for ft2 in fraction_test:                            
            indices_test[ft2] = True

        
        data.train_mask = torch.tensor(indices_train)
        data.val_mask = torch.tensor(indices_val)
        data.test_mask = torch.tensor(indices_test)
    else:
        data.test_mask = torch.tensor([i for i in range(nv)])

    return data

def graphListToData(As, Xs, Ys, NXs, test=False):
    # As List of Adjacency matrices
    # Xs List of features
    # Ys List of node labels
    counter_protein = 0
    indices_protein = []
    for x in Xs:
        indices_protein.append([counter_protein, counter_protein + x.shape[0]])
        counter_protein += x.shape[0]
    assert(len(As) == len(Xs))
    assert(len(As) == len(Ys))
    data_list = [graphToData(A, X, Y, NX, test=test) for A, X, Y, NX in zip(As, Xs, Ys, NXs)]
    data_batches = Batch.from_data_list(data_list)
    return data_batches, indices_protein


def computeDistanceResidue(residue1, residue2, residue_to_atoms, mode="min"):
    if mode == "min":
        atoms_residue1 = residue_to_atoms[residue1]
        atoms_residue2 = residue_to_atoms[residue2]
        coordinates1 = [x.get_coord() for x in atoms_residue1]
        coordinates2 = [x.get_coord() for x in atoms_residue2]
        distance_r12 = min(scipy_distance.cdist(coordinates1, coordinates2).min(axis=1))
    return distance_r12




def loadGraphFromFile(filename, name_to_pattern, classification_type="helices", distance_based=True, distance_threshold=2, nb_neighbors=2, nb_features=4, shuffle=False, nmr_conformations=False):
    # Returns protein graph in format A: adjacency matrix, X: node features matrix, Y node labels
    # filename: String, name of file in mmCIF format
    # name_to_pattern: Dictionary .cif name to pattern (str to str) 
    # classification_type: Str: "helices", "strand", "helices_strands" or "all"
    # distance_based: Boolean, if True then returns graph constructed based on distance_threshold. If false uses a sequence graph of the protein sequence
    # distance_threshold: Float/int, In the distance_based mode, distance (in Angström) after which no edge is considered
    # num_features: Integer, 2 or 4: cosinus of dihedral angles phi/psi or cosinus/sinus of same angles respectively
    parser = MMCIFParser(QUIET=True)
    try:
        protein_filename = filename
        structure = parser.get_structure(protein_filename, protein_filename)
        dssp=PDB.DSSP(structure[0], protein_filename)
    except:
        print("\nCould not parse or no file named %s" %protein_filename)
        return [], [], [], [], []
    
    # Two scales: 
    # - Atom level
    # - Residue level
    
    # 1 - Atom level
    # a - distances between atoms
    # b - atom to residue? / residue to atom

    # 2 - Residue level
    # a - residue to dihedral angles in np.array() shape
    # b - adjencency matrix in np.array() shape
    # c - residue labels


    # I - Get list of secondary structures computed with DSSP
    # WARNING, DSSP might return less keys (each key correspond to a residue) than the total number of residues in the protein
    #   H 	Alpha helix (4-12)
    #   B 	Isolated beta-bridge residue
    #   E 	Strand
    #   G 	3-10 helix
    #   I 	Pi helix
    #   T 	Turn
    #   S 	Bend
    #   - 	None

    X_phi_psi = []                     
    X_omega = []
    Y_labels = []                                     
    debug_Y = []
    features_FOS = []
    
    map_st = {"G": 0, "H": 1, "I": 2, "T": 3, "E": 4, "B": 5, "S": 6, "-": 7}  # maps secondary structure labels to integers
    
    #0 	DSSP index
    #1 	Amino acid
    #2 	Secondary structure
    #3 	Relative ASA
    #4 	Phi
    #5 	Psi
    #6 	NH-->O_1_relidx
    #7 	NH-->O_1_energy
    #8 	O-->NH_1_relidx
    #9 	O-->NH_1_energy
    #10 	NH-->O_2_relidx
    #11 	NH-->O_2_energy
    #12 	O-->NH_2_relidx
    #13 	O-->NH_2_energy
    
    residue_shapes = []
    residue_labels = []  # 0: alpha_helix, 1: beta_strand, 2: other
    residue_all_labels = []
    residue_keys = []
    residue_dssp = []
    x_phi_psi = []
 
    for key in dssp.keys():
        residue_key = dssp[key]
        residue_dssp.append(residue_key)
        residue_keys.append(residue_key[2])
        residue_all_labels.append(map_st[residue_key[2]])
        x_phi = residue_key[4]
        x_psi = residue_key[5]
        x_phi_psi.append([x_phi, x_psi])
        residue_labels.append(map_st[residue_key[2]])
        if 1:
            residue_shapes.append(residue_key)
            debug_Y.append(residue_key[2])
    
    y_label = residue_labels[:-1] 
    x_phi_ = [x[0] for x in x_phi_psi][1:]
    x_psi_ = [x[1] for x in x_phi_psi][:-1]
    x_phi_psi = [[x,y] for x,y in zip(x_phi_, x_psi_)] 
    Y_labels.append(y_label)
    X_phi_psi.append(x_phi_psi)
    assert(len(x_phi_psi) == len(y_label))
 
    # II - Get 3D coordinates
    coordinates = [x.get_coord() for x in structure.get_atoms()]
    atoms_full_ids = [x.get_full_id() for x in structure.get_atoms()]

    
    residue_to_atoms = {}
    atom_to_residue = {}
    atom_to_coordinates = {}
    
    for i, a in enumerate(structure.get_atoms()):
        atom_to_coordinates[a] = a.get_coord() 

    for residue in structure.get_residues():
        residue_to_atoms[residue] = []
        for x in residue.get_atoms():
            residue_to_atoms[residue].append(x)
            atom_to_residue[x] = residue
    
    # graph of atoms
    # graph of residues

    all_patterns = False    
    all_residues = [residue for residue in structure.get_residues()]
    if len(name_to_pattern) > 0:
        pattern_to_keep = name_to_pattern[protein_filename.split("/")[-1].split(".cif")[0]]

    else:
        pattern_to_keep = [residue.get_full_id()[2] for residue in all_residues]
        all_patterns = True

    #print(pattern_to_keep[0])
    # NEED TO JOIN WITH DSSP
    keys = [key for key in dssp.keys()]

    ids_to_keep = []
    for key in keys:
        id_to_keep = key[0]  + " " + " ".join([str(ke) for ke in key[1]]) 
        ids_to_keep.append(id_to_keep)

    full_ids = []
    for x in all_residues:
        xx = x.get_full_id()[2:]
        full_id = xx[0]  + " " + " ".join([str(ke) for ke in xx[1]])
        full_ids.append(full_id)
    
    residues = []
    for i, residue in enumerate(all_residues):
        if full_ids[i] in ids_to_keep:
            pattern_residue = residue.get_full_id()[2]
            if not all_patterns:
                if pattern_residue == pattern_to_keep:
                    residues.append(residue)
            else:
                residues.append(residue)


    final_residues = []
    if nmr_conformations:
        for i in range(len(residues)):
            if residues[i].get_full_id()[1] == 0:
                final_residues.append(residues[i])
            else:
                break
        residues = final_residues

    if shuffle:
        random.shuffle(residues)

    nb_residues = len(residues)
    X = np.zeros((nb_residues, 1))
    Y = np.zeros((nb_residues,))
    NX = []

    
    A_data = []
    A_cols = []
    A_row1 = []
    A_row2 = []

    window_width = 2 
    ratio_conversion = np.pi/180  
    distances = {}
    standard_amino_acids_map = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'PYL': 15, 'SEC': 16, 'SER': 17, 'THR': 18, 'TRP': 19, 'TYR': 20, 'VAL': 21, 'UNK': 22}

    for i in range(len(residues)-1):
        nb_remaining = len(residues) - i - 1
        # Based on distances
        distances_i = []
        #for j in range(i+1, len(residues)):
        for j in range(len(residues)-1):
            if i != j:
                residue_i = residues[i]
                residue_j = residues[j]
                dij = computeDistanceResidue(residue_i, residue_j, residue_to_atoms)
                distances_i.append(dij)
            else:
                distances_i.append(1e+10)
        
        indices_closest_neighbors = np.array(distances_i).argsort()[:nb_neighbors].tolist()
        #print(sorted(distances_i)[:10])
        #pdb.set_trace()
        X[i] = standard_amino_acids_map[residues[i].resname]
        ratio_conversion = np.pi/180.
        residue_key = dssp[residues[i].get_full_id()[2:]]
        x_phi = ratio_conversion*residue_key[4] 
        x_psi = ratio_conversion*residue_key[5]
        features_FOS_i = [np.cos(x_phi), np.cos(x_psi)]

        FOS_i = []
        for inj, nj in enumerate(indices_closest_neighbors): 
            # Compute dihedral angle between residue i and residue j
            # Angles is already in radians, on the contrary of DSSP where angles are in degrees
            phi_ij = calc_dihedral(residues[i]['C'].get_vector(),residues[nj]['N'].get_vector(),residues[nj]['CA'].get_vector(), residues[nj]['C'].get_vector())
            # C(i-1),N(i),Ca(i),C(i)
            psi_ij = calc_dihedral(residues[i]['N'].get_vector(),residues[i]['CA'].get_vector(),residues[i]['C'].get_vector(), residues[nj]['N'].get_vector())
            # N(i),Ca(i),C(i),N(i+1)

            edge_ij_features = [np.cos(phi_ij), np.cos(psi_ij)]
            A_row1.append(i)
            A_row2.append(nj)
            NX.append(edge_ij_features)
            A_row1.append(nj)
            A_row2.append(i)
            NX.append(edge_ij_features)

            FOS_i.append(edge_ij_features)

        mean_phi_i, mean_psi_i = np.mean([x[0] for x in FOS_i]), np.mean([x[1] for x in FOS_i])
        std_phi_i, std_psi_i = np.std([x[0] for x in FOS_i]), np.std([x[1] for x in FOS_i])
        features_FOS_i += [mean_phi_i, std_phi_i, mean_psi_i, std_psi_i]
        features_FOS.append(features_FOS_i)
        
        residue_key = dssp[residues[i].get_full_id()[2:]]
        Y[i] = map_st[residue_key[2]]
    

    Y = Y.astype(int)
    A = [A_row1, A_row2]
    return A, X, Y, NX, features_FOS


import time
import sys
import threading



def loadGraphsFromDirectory(filedir, name_to_pattern, classification_type="helices", distance_based=True, distance_threshold=5, nb_neighbors=2, nb_features=4, shuffle=False, nmr_conformations=False):
    # Returns list of proteins graph, each in format A: adjacency matrix, X: node features matrix, Y node labels
    # filename: name of file in mmCIF format
    # name_to_pattern: Dictionary .cif name to pattern (str to str) 
    # classification_type: Str: "helices", "strand", "helices_strands" or "all"
    # distance_based: Boolean, if True then returns graph constructed based on distance_threshold. If false uses a sequence graph of the protein sequence
    # distance_threshold: In the distance_based mode, distance (in Angström) after which no edge is considered
    # num_features: Integer, 2 or 4: cosinus of dihedral angles phi/psi or cosinus/sinus of same angles respectively
    As = []
    Xs = []
    Ys = []
    NXs = []
    FOSs = []
    filew = open("listPDB1.txt", "w")
    for subdir, _, _ in os.walk(filedir):
        if subdir != filedir:
            for _, _, files in os.walk(subdir):
                for i_protein, protein_filename in enumerate(files):
                    if '.cif' in protein_filename:
                        sys.stdout.write("\rProtein filename, number %i" % i_protein)
                        sys.stdout.flush()
                        A, X, Y, NX, FOS = loadGraphFromFile(subdir + "/"  + protein_filename, name_to_pattern, classification_type=classification_type, distance_based=distance_based, distance_threshold=distance_threshold, nb_neighbors=nb_neighbors, nb_features=nb_features, shuffle=shuffle, nmr_conformations=nmr_conformations)
                        if len(A) > 0:
                            As.append(A)
                            Xs.append(X)
                            Ys.append(Y)
                            NXs.append(NX)
                            FOSs.append(FOS)
                            filew.write(protein_filename.split('.cif')[0] + "\n")
                        
    filew.close()
    return As, Xs, Ys, NXs, FOSs




if __name__ == "__main__":
    import pdb
    import sys
    filename = sys.argv[1]
    A, X, Y, NX = loadGraphFromFile(filename)
    pdb.set_trace()

