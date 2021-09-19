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
import pdb, os, sys, random, time
import sequoia_clifford_algebra

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
        nb_train = int(29*nv/40)
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




def graphListToDataList(As, Xs, Ys, NXs, test=False):
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
    data_list = [[A, X, Y, NX] for A, X, Y, NX in zip(As, Xs, Ys, NXs)]
    return data_list, indices_protein

def returnDataFromDataList(A, X, Y, NX):
    A = torch.tensor(A, dtype=torch.long)
    X = torch.tensor(X, dtype=torch.float)
    NX = torch.tensor(NX, dtype=torch.float)
    data = Data(edge_index=A, x=X, edge_attr=NX)
    data.y = torch.tensor(Y)

    nv =Y.shape[0]
    
    mask_tmp = [i for i in range(nv)]
    random.shuffle(mask_tmp)
    nb_train = int(29*nv/40)
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



    return data


def computeDistanceResidue(residue1, residue2, residue_to_atoms, mode="min", calpha_mode=False):
    if mode == "min":
        if not calpha_mode:
            atoms_residue1 = residue_to_atoms[residue1]
            atoms_residue2 = residue_to_atoms[residue2]
            coordinates1 = [x.get_coord() for x in atoms_residue1]
            coordinates2 = [x.get_coord() for x in atoms_residue2]
            distance_r12 = min(scipy_distance.cdist(coordinates1, coordinates2).min(axis=1))

        if calpha_mode:
            calpha_residues1 = [atom for atom in residue_to_atoms[residue1] if "CA" in atom.fullname]
            calpha_residues2 = [atom for atom in residue_to_atoms[residue2] if "CA" in atom.fullname]
            if len(calpha_residues1) == 0:
                calpha_residue1 = [atom for atom in residue_to_atoms[residue1]][0]
            else:
                calpha_residue1 = calpha_residues1[0]

            if len(calpha_residues2) == 0:
                calpha_residue2 = [atom for atom in residue_to_atoms[residue2]][0]
            else:
                calpha_residue2 = calpha_residues2[0]



            coordinates1 = calpha_residue1.get_coord()
            coordinates2 = calpha_residue2.get_coord()
            distance_r12 = np.linalg.norm(coordinates1-coordinates2)

    return distance_r12


def dataAugmentation(As, Xs, Ys, NXs, nb_labels, map_st, overload=False):
    if len(Ys[0]) > 0:
        ground_truth_provided = True
        Ys = [Y_[:-1] for Y_ in Ys]
        #Y_values = sum(Y_values, [])
        #Y_values = sorted(list(set(Y_values)))
        #Y_tmp = []
        #Y_map = {}
        #for i, vy in enumerate(Y_values):
        #    Y_map[vy] = i
        #Ys = [np.array([map_st[y] for y in Y]) for Y in Ys]
    else:
        ground_truth_provided = False
        Ys = [np.array([lab for lab in range(nb_labels-1)] + [nb_labels-1 for i in range(len(Xs[0]) - nb_labels)]).astype(int)]

    #############################################################################
    ########################## Data augmentation ###############################
    X_phi_psi_0 = []
    #Ys = [np.array([map_st[yy] for yy in y.tolist()]) for y in Ys]
    X_phi_psi_0 = []
    for i_PA, PA in enumerate(As):
        X_phi_psi_0_D = {}
        for i_edge, source in enumerate(PA[0]):
            if not overload:
                if i_edge % 2 == 0:
                    feature_source = NXs[i_PA][i_edge]
                    current_feature_source = X_phi_psi_0_D.get(source, []) 
                    current_feature_source += feature_source 
                    X_phi_psi_0_D[source] = current_feature_source
            else:
                feature_source = NXs[i_PA][i_edge]
                current_feature_source = X_phi_psi_0_D.get(source, []) 
                current_feature_source += feature_source 
                X_phi_psi_0_D[source] = current_feature_source

            
        X_phi_psi_P = [[] for key in X_phi_psi_0_D]
        for key in X_phi_psi_0_D:
            X_phi_psi_P[key] = X_phi_psi_0_D[key]
        X_phi_psi_0.append(X_phi_psi_P)

    
    try:
        Xs = [np.concatenate([X_[:-1], X_phi_psi_0[i]], axis=1) for i, X_ in enumerate(Xs)]
    except:
        Xs = [np.concatenate([Xs[0], np.array(X_phi_psi_0[0])], axis=1)]
    #############################################################################
    return As, Xs, Ys, NXs, ground_truth_provided 


def sac_map(calpha_mode):
    if not calpha_mode:
        standard_amino_acids_map = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6,\
                                    'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE':\
                                    13, 'PRO': 14, 'PYL': 15, 'SEC': 16, 'SER': 17, 'THR': 18, 'TRP': 19,\
                                    'TYR': 20, 'VAL': 21, 'UNK': 22, 'MSE': 23}
        # normalize features
        max_value = max(standard_amino_acids_map.values())
        for key in standard_amino_acids_map.keys():
            standard_amino_acids_map[key] /= max_value

    else:
        standard_amino_acids_map = {'ALA': 0, 'ARG': 0, 'ASN': 0, 'ASP': 0, 'CYS': 0, 'GLN': 0, 'GLU': 0,\
                                    'GLY': 0, 'HIS': 0, 'ILE': 0, 'LEU': 0, 'LYS': 0, 'MET': 0, 'PHE': 0,\
                                    'PRO': 0, 'PYL': 0, 'SEC': 0, 'SER': 0, 'THR': 0, 'TRP': 0, 'TYR': 0,\
                                    'VAL': 0, 'UNK': 0, 'MSE': 0}
    return standard_amino_acids_map




def loadGraphFromFile(filename, name_to_pattern, classification_type="helices", distance_based=True, distance_threshold=2, nb_neighbors=2, nb_features=4, shuffle=False, nmr_conformations=False, calpha_mode=False, dssp_mode=False):
    # Returns protein graph in format A: adjacency matrix, X: node features matrix, Y node labels, NX edge features
    # filename: String, name of file in pdb or mmCIF format
    # name_to_pattern: Dictionary .cif name to pattern (str to str) 
    # classification_type: Str: "helices", "strand", "helices_strands" or "all"
    # distance_based: Boolean, if True then returns graph constructed based on distance_threshold. If false uses a sequence graph of the protein sequence
    # distance_threshold: Float/int, In the distance_based mode, distance (in Angström) after which no edge is considered
    # num_features: Integer, 2 or 4: cosinus of dihedral angles phi/psi or cosinus/sinus of same angles respectively
    
    if ".cif" in filename:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    try:
        protein_filename = filename
        structure = parser.get_structure(protein_filename, protein_filename)
        #if not calpha_mode:
        if dssp_mode:
            dssp=PDB.DSSP(structure[0], protein_filename)
    except:
        print("\nCould not parse or no file named %s" %protein_filename)
        return [], [], [], [], []
    
    map_st = {"G": 0, "H": 1, "I": 2, "T": 3, "E": 4, "B": 5, "S": 6, "-": 7}  # map used for secondary structure labels


    

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
    
    # graph of residues

    all_patterns = False    
    all_residues = [residue for residue in structure.get_residues()]
    if len(name_to_pattern) > 0:
        pattern_to_keep = name_to_pattern[protein_filename.split("/")[-1].split(".cif")[0]]

    else:
        pattern_to_keep = [residue.get_full_id()[2] for residue in all_residues]
        all_patterns = True


    ########################################################################################################################
    ################################################### JOIN WITH DSSP #####################################################
    full_ids = []
    for x in all_residues:
        xx = x.get_full_id()[2:]
        full_id = xx[0]  + " " + " ".join([str(ke) for ke in xx[1]])
        full_ids.append(full_id)

    if not dssp_mode:
        ids_to_keep = full_ids.copy()
    else:
        keys = [key for key in dssp.keys()]
        ids_to_keep = []
        for key in keys:
            id_to_keep = key[0]  + " " + " ".join([str(ke) for ke in key[1]]) 
            ids_to_keep.append(id_to_keep)

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
    ########################################################################################################################
    ########################################################################################################################

    ########################################################################################################################
    ################################################## Graph construction ##################################################

    nb_residues = len(residues)
    X = np.zeros((nb_residues, 1))
    Y = np.zeros((nb_residues,))      

    standard_amino_acids_map = sac_map(calpha_mode) 

    NX = []
    DISTANCES = []
    A_row1 = []
    A_row2 = []

    for i in range(len(residues)-1):
        nb_remaining = len(residues) - i - 1
        distances_i = []
        for j in range(len(residues)-1):
            if i != j:
                residue_i = residues[i]
                residue_j = residues[j]
                dij = computeDistanceResidue(residue_i, residue_j, residue_to_atoms, calpha_mode=calpha_mode)
                distances_i.append(dij)
            else:
                distances_i.append(1e+10)
        DISTANCES.append(distances_i) 

    DISTANCES = np.array(DISTANCES)
    indices_closest_neighbors = np.argsort(DISTANCES, axis=1)

    for i in range(len(residues)-1):
        nb_remaining = len(residues) - i - 1
        sys.stdout.write("\r%i residues remaining" % nb_remaining)
        sys.stdout.flush() 
        # Based on distances
        if distance_based:
            distances_i = DISTANCES[i]
            indices_closest_neighbors_i = indices_closest_neighbors[i]
            X[i] = standard_amino_acids_map[residues[i].resname]
            FOS_i = []
            for inj, nj in enumerate(indices_closest_neighbors_i[:nb_neighbors]): 
                distances_nj = DISTANCES[nj]
                if not calpha_mode:
                    # Compute dihedral angle between residue i and residue j
                    # Angles is already in radians, on the contrary of DSSP where angles are in degrees
                    phi_ij = calc_dihedral(residues[i]['C'].get_vector(),residues[nj]['N'].get_vector(),residues[nj]['CA'].get_vector(), residues[nj]['C'].get_vector())
                    # C(i-1),N(i),Ca(i),C(i)
                    psi_ij = calc_dihedral(residues[i]['N'].get_vector(),residues[i]['CA'].get_vector(),residues[i]['C'].get_vector(), residues[nj]['N'].get_vector())
                    # N(i),Ca(i),C(i),N(i+1)
                    edge_ij_features = [np.cos(phi_ij), np.cos(psi_ij)]

                else:
                    Calpha_i = i
                    Calpha_j = nj
                    # Compute dihedral angle between residue i and residue j
                    # Angles is already in radians, on the contrary of DSSP where angles are in degrees
                    # CA(iprime) = l'atome CA le plus proche de CA(i) et different de CA(j)
                    # CA(jprime) =  l'atome CA le plus proche de CA(j) et different de CA(i) et CA(iprime)
                    # Et ensuite, on calcule l'angle diedre entre: CA(iprime)-CA(i)-CA(j)-CA(jprime).
                    if inj == 0:
                        indices_closest_neighbors_nj = indices_closest_neighbors[nj]
                        Calpha_iprime = indices_closest_neighbors_i[1] 
                        inj_prime = 0
                        while indices_closest_neighbors_nj[inj_prime] == i or indices_closest_neighbors_nj[inj_prime] == Calpha_iprime: inj_prime += 1
                        Calpha_jprime = indices_closest_neighbors_nj[inj_prime]
                        Calphas = [Calpha_iprime, Calpha_i, Calpha_j, Calpha_jprime]
                    else: # second case: use average distance between i and nj to decide iprime or jprime.
                        indices_closest_neighbors_nj = indices_closest_neighbors[nj]
                        prime_candidates_i = [indices_closest_neighbors_i[xi] for xi in range(min(3, len(indices_closest_neighbors_i))) if xi != 1] 
                        prime_candidates_j = [indices_closest_neighbors_nj[xj] for xj in range(min(len(indices_closest_neighbors_nj),10)) if indices_closest_neighbors_nj[xj] != i] 
                    
                        prime_candidates = prime_candidates_i
                        total_distance_candidate_0 = distances_i[prime_candidates[0]] + distances_nj[prime_candidates[0]]
                        total_distance_candidate_1 = distances_i[prime_candidates[1]] + distances_nj[prime_candidates[1]]
                        if total_distance_candidate_0 < total_distance_candidate_1:
                            Calpha_iprime = prime_candidates[0]
                        else:
                            Calpha_iprime = prime_candidates[1]
                    
                        Calpha_jprime = [pcj for pcj in prime_candidates_j if pcj != Calpha_iprime and pcj != i][0] 
                        Calphas = [Calpha_iprime, Calpha_i, Calpha_j, Calpha_jprime]
                    
                    x = DISTANCES[Calphas[0], Calphas[1]]
                    y = DISTANCES[Calphas[1], Calphas[3]]
                    a = DISTANCES[Calphas[0], Calphas[3]]
                    b = DISTANCES[Calphas[1], Calphas[2]] 
                    c = DISTANCES[Calphas[0], Calphas[2]]
                    d = DISTANCES[Calphas[2], Calphas[3]]
                    
                    D_i_noisy = np.array([[0, x, c, a], [x, 0, b, y], [c, b, 0, d], [a, y, d, 0]])
                    D_i = sequoia_clifford_algebra.projectMatrixPosDef(D_i_noisy)
                    
                    x_, y_, a_, b_, c_, d_ = D_i[0,1], D_i[1,3], D_i[0,3], D_i[1,2], D_i[2,0], D_i[3,2]
                    cos_phi_ij = sequoia_clifford_algebra.compute_formula(x_, y_, a_, b_, c_, d_) 
                    
                    edge_ij_features = [cos_phi_ij, distances_i[nj]]


                A_row1.append(i)
                A_row2.append(nj)
                NX.append(edge_ij_features)
                A_row1.append(nj)
                A_row2.append(i)
                NX.append(edge_ij_features)

        if dssp_mode:     
            residue_key = dssp[residues[i].get_full_id()[2:]]
            Y[i] = map_st[residue_key[2]]
    
    if dssp_mode:
        Y = Y.astype(int)
    else:
        Y = []

    A = [A_row1, A_row2]
    ########################################################################################################################
    ########################################################################################################################
    
    ###################################################
    ############# Recaling distances ##################
    distances_max_protein = max([nx[1] for nx in NX])
    NX_tmp = []
    for nx in NX:
        rescaled_distance = nx[1]/distances_max_protein
        NX_tmp.append([nx[0], rescaled_distance])
    NX = NX_tmp
    ###################################################

    return A, X, Y, NX, [] 



def loadGraphsFromDirectory(filedir, name_to_pattern, classification_type="helices", distance_based=True, distance_threshold=5, nb_neighbors=2, nb_features=4, shuffle=False, nmr_conformations=False, calpha_mode=False, dssp_mode=True):
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
    for subdir, _, _ in os.walk(filedir):
        if subdir != filedir:
            for _, _, files in os.walk(subdir):
                for i_protein, protein_filename in enumerate(files):
                    if '.cif' in protein_filename:
                        sys.stdout.write("\rProtein filename, number %i" % i_protein)
                        sys.stdout.flush()
                        A, X, Y, NX, [] = loadGraphFromFile(subdir + "/"  + protein_filename, name_to_pattern, classification_type=classification_type, distance_based=distance_based, distance_threshold=distance_threshold, nb_neighbors=nb_neighbors, nb_features=nb_features, shuffle=shuffle, nmr_conformations=nmr_conformations, calpha_mode=calpha_mode, dssp_mode=dssp_mode)
                        if len(A) > 0:
                            As.append(A)
                            Xs.append(X)
                            Ys.append(Y)
                            NXs.append(NX)
                        
    filew.close()
    return As, Xs, Ys, NXs, FOSs


if __name__ == "__main__":
    import pdb
    import sys
    filename = sys.argv[1]
    A, X, Y, NX = loadGraphFromFile(filename)
    pdb.set_trace()

