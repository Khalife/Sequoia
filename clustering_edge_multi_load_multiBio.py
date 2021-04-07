import numpy as np
from scipy.sparse import coo_matrix
import torch
import igraph
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
    #At = dense_to_sparse(torch.tensor(A)) #, dtype=torch.long))
    At = torch.tensor(A, dtype=torch.long)
    Xt = torch.tensor(X, dtype=torch.float)
    NXt = torch.tensor(NX, dtype=torch.float)
    #data = Data(edge_index=At[0], x=Xt) 
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
        for (ft1, fv, ft2) in zip(fraction_train, fraction_val, fraction_test):
            indices_train[ft1] = True
            indices_val[fv] = True 
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


def ssFileToTable(ss_filename, pattern, keep_all = False):
    table = {}
    all_cluster_types = []
    with open(ss_filename, "r") as f:
        for line in f:
            line_split = line.split()
            if keep_all:
                residue_number = int(line_split[1].split(":")[0])
                residue_cluster = " ".join(line_split[2:]) # ~.split(":")[0]
                all_cluster_types.append(line_split[2])
                table[residue_number] = residue_cluster
            else:
                if line_split[1].split(":")[1] == pattern:
                    residue_number = int(line_split[1].split(":")[0])
                    residue_cluster = " ".join(line_split[2:]) # ~.split(":")[0]
                    all_cluster_types.append(line_split[2])
                    table[residue_number] = residue_cluster



    
    cluster_types = set([])
    cluster_types.update(all_cluster_types)
    return cluster_types, table





def loadGraphFromFile(filename, ss_filename, name_to_pattern, nb_neighbors, classification_type="helices", distance_based=True, distance_threshold=2, nmr_conformations=False, shuffle=False):
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
        #return [], [], [], [], [], [], []
    except:
        #print("\nCould not parse or no file named %s" %protein_filename)
        data_filename = protein_filename.split("/")[-1]
        #print(data_filename)
        return [], [], [], [], [], [], [], [], [], []
    
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
    
    if classification_type == "all": 
        map_st = {"G": 0, "H": 1, "I": 2, "T": 3, "E": 4, "B": 5, "S": 6, "-": 7}
    if classification_type == "helices_strands":
        map_st = {"G": 0, "H": 0, "I": 0, "T": 2, "E": 1, "B": 2, "S": 2, "-":  2}
    # Binary class: detection of alpha helices
    if classification_type == "helices":
        map_st = {"G": 0, "H": 0, "I": 0, "T": 1, "E": 1, "B": 1, "S": 1, "-": 1 }
    # Binary class: detection of beta strands
    if classification_type == "strands":
        map_st = {"G": 0, "H": 0, "I": 0, "T": 0, "E": 1, "B": 0, "S": 0, "-": 0 }


    
    
    edges = []
    nodes = []
    G = igraph.Graph()
    
    #alpha_helix= ["H", "G", "I"]
    #beta_strand = ["E"]
    
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
    
    # OTHER WAY TO GET THE RESIDUE TYPES: [x.xtra for x in structure.get_residues() if len(x.xtra) > 0]
    # This should have the same length as the list obtained from dssp
    residue_shapes = []
    residue_labels = []  # 0: alpha_helix, 1: beta_strand, 2: other
    residue_all_labels = []
    residue_keys = []
    residue_dssp = []
    x_phi_psi = []
    #if len(dssp.keys()) > 100:
    #    debug_counter += 1
 
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
        #residue_id = " ".join([str(sr) for sr in residue.get_full_id()])
        for x in residue.get_atoms():
            #x_id = " ".join([str(sx) for sx in x.get_full_id()])
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

    if nmr_conformations:
        cluster_types, ssTable = ssFileToTable(ss_filename, pattern_to_keep[0], keep_all=True)
    else:
        cluster_types, ssTable = ssFileToTable(ss_filename, pattern_to_keep)

    cluster_types = sorted(cluster_types)


    #print(pattern_to_keep[0])
    
    # NEED TO JOIN WITH DSSP
    keys = [key for key in dssp.keys()]

    ids_to_keep = []
    for key in keys:
        #id_to_keep = key[0]  + " " + " ".join([str(ke) for ke in key[1]]) 
        id_to_keep = key[0]  + " " + " ".join([str(ke) for ke in key[1][1:]]) 
        ids_to_keep.append(id_to_keep)

    full_ids = []
    for x in all_residues:
        xx = x.get_full_id()[2:]
        #full_id = xx[0]  + " " + " ".join([str(ke) for ke in xx[1]])
        full_id = xx[0]  + " " + " ".join([str(ke) for ke in xx[1][1:]])
        full_ids.append(full_id)
    
    residues = []
    for i, residue in enumerate(all_residues):
        if full_ids[i] in ids_to_keep:
        #if 1:
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
    #A = np.zeros((nb_residues, nb_residues))
    nb_features = 2
    #nb_neighbors = 2
    nb_total_features = nb_features*nb_neighbors + 1
    X = np.zeros((nb_residues, 1))
    Y = np.zeros((nb_residues,))
    NX = []
    
    

    edges_distance_attribute = []
    
    A_data = []
    A_cols = []
    A_row1 = []
    A_row2 = []

    window_width = 2 
    ratio_conversion = np.pi/180  
    distances = {}
    #standard_amino_acids_map = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 'PYL': 15, 'SEC': 16, 'SER': 17, 'THR': 18, 'TRP': 19, 'TYR': 20, 'VAL': 21, 'UNK': 22, 'KCX': 23, 'MSE': 24, 'LLP': 25, 'CSD': 26, 'NLE': 27, 'PCA': }

    nodes_attributes = [] #residue.get_full_id() for residue in residues]
    sec_types = []
    hashTable_D = {}

    residues_beta_index = []
    for ires, residue in enumerate(residues):
        try:
            sec_type = ssTable[residue.get_full_id()[3][1]]
            if "sheet" in sec_type:
                residues_beta_index.append(ires)
        except:
            continue

    if len(residues_beta_index) == 0:
        return [], [], [], [], [], [], [], [] ,[], []

    #distances_edges = np.zeros((nb_residues, nb_residues))
    distances_edges = (1e+10)*np.ones((len(residues_beta_index), len(residues_beta_index)))



    for i in range(len(residues)):
        # if residues_i is beta
        try:
            sec_type = ssTable[residues[i].get_full_id()[3][1]]

            if "sheet" in sec_type:
                sec_type = "E"

            #if "sheet" in sec_type:
            #    sec_types.append("E")
            #else:
            #    sec_types.append("H")
            
        except:
            #sec_types.append("other")
            sec_type = "other"



        #if 1
        if sec_type == "E":
            residue_i = residues[i]
            sec_types.append("E")
            nodes_attributes.append(residues[i].get_full_id())

            try:
                residue_key = dssp[residues[i].get_full_id()[2:]]
            except:
                try:
                    residue_key = dssp[(residues[i].get_full_id()[2:][0], (" ", residues[i].get_full_id()[2:][1][1], residues[i].get_full_id()[2:][1][2]))]
                except:
                    pdb.set_trace()

            #X[i] = standard_amino_acids_map[residues[i].resname]
            X[i] = 0 #float(residues[i].resname)
            Y[i] = map_st[residue_key[2]]
            
            
            
            
            
            
            
            
            

            #Y[i] = map_st[residue_key[2]]
            #if residue_key[2] == classification_type:
            if i < len(residues)-1:
                 
                nb_remaining = len(residues) - i - 1
                #sys.stdout.write("\r%i residues remaining" % nb_remaining)
                #sys.stdout.flush() 
                # Based on distances
                dij = 0
                if distance_based:
                    distances_i = []
                    #for j in range(i+1, len(residues)):
                    residues_i_considered = []
                    for j in range(len(residues)):
                        try:
                            sec_type_j = ssTable[residues[j].get_full_id()[3][1]]
                        except:
                            sec_type_j = "other" 
                        if "sheet" in sec_type_j: 
                            residue_j = residues[j]
                            if i != j:
                                #if i == 3:
                                #    pdb.set_trace()
                                dij = computeDistanceResidue(residue_i, residue_j, residue_to_atoms)
                                distances_i.append(dij)
                            else:
                                dij = 1e+10
                                distances_i.append(dij)

                            residues_i_considered.append(residue_j)
                            
                            #distances_edges[i, j] = dij
                            i_beta = residues_beta_index.index(i) 
                            j_beta = residues_beta_index.index(j) 
                            distances_edges[i_beta, j_beta] = dij
                            name_i = str(residues[i].get_full_id()[3][1])
                            name_j = str(residues[j].get_full_id()[3][1])
                            hashTable_D[" ".join(sorted([name_i, name_j]))] = dij

                    #if i == 3:
                    #    pdb.set_trace()
                    #distance_threshold = 2.92 # Angstrom
                    #nb_neighbors = len([x for x in distances_i if x <= distance_threshold])
                    #pdb.set_trace()
                    nb_neighbors_false = 10
                    indices_closest_neighbors = np.array(distances_i).argsort()[:nb_neighbors_false].tolist()
                    ratio_conversion = np.pi/180.
                    x_phi = ratio_conversion*residue_key[4] 
                    x_psi = ratio_conversion*residue_key[5]
                    features_FOS_i = [np.cos(x_phi), np.cos(x_psi)]

                    
                    FOS_i = []
                    for inj, nj in enumerate(indices_closest_neighbors):

                        phi_ij = 0
                        psi_ij = 0

                        edge_ij_features = [np.cos(phi_ij), np.cos(psi_ij)]
                        

                        try:
                            i_beta = residues_beta_index.index(i) 
                            nj_beta = nj 
                        except:
                            pdb.set_trace()

                        A_row1.append(i_beta)
                        A_row2.append(nj_beta)
                        NX.append(edge_ij_features)
                        A_row1.append(nj_beta)
                        A_row2.append(i_beta)
                        NX.append(edge_ij_features)

                        FOS_i.append(edge_ij_features)
                        edges.append([i_beta, nj_beta])
                        edges_distance_attribute.append(distances_edges[i_beta, nj_beta])


                    mean_phi_i, mean_psi_i = np.mean([x[0] for x in FOS_i]), np.mean([x[1] for x in FOS_i])
                    std_phi_i, std_psi_i = np.std([x[0] for x in FOS_i]), np.std([x[1] for x in FOS_i])
                    features_FOS_i += [mean_phi_i, std_phi_i, mean_psi_i, std_psi_i]
                    features_FOS.append(features_FOS_i)
                            
                

                #sec_types.append(residue_key[2])
    

    Y = Y.astype(int)
    #A = coo_matrix((A_data, (A_rows, A_cols)), shape=(2, len(A_data)))
    A = [A_row1, A_row2]
    G.add_vertices(len(nodes_attributes))
    G.vs["name"] = nodes_attributes
    G.vs["sec_type"] = sec_types
    G.add_edges(edges)
    G.es["distance"] = edges_distance_attribute 
    #pdb.set_trace()
    #for cluster_type in cluster_types:


    strands_OUTPUTS = []
    SHEETS_outputs = []
    H_scores = []
    i_threshold_counter = -1
    for threshold_G1 in np.linspace(1.1, 2, 10).tolist():
        for threshold_GB in np.linspace(1.5, 3.5, 10).tolist():
            i_threshold_counter += 1
            
            G0 = G.subgraph([x for x in G.vs.select(lambda x: x["sec_type"] == "E")])
            G1 = igraph.Graph()
            G1_distances = []
            G1_edges = []
            for vs in G0.vs:
                neis = list(set(G0.neighbors(vs)))
                distances_vs = [G0.es[G0.get_eid(vs, nei)]["distance"] for nei in neis] 
                to_keep_neighbors = np.argsort(distances_vs)[:nb_neighbors].tolist()
                if 1:
                    #threshold_G1 = 1.65 # Angstom
                    #threshold_G1 = 1.2 # Angstrom
                    to_keep_neighbors = [index_nei for index_nei in to_keep_neighbors if distances_vs[index_nei] <= threshold_G1] 
                G1_edges += [(vs.index, neis[nei_index]) for nei_index in to_keep_neighbors]
                G1_distances += [distances_vs[tkn] for tkn in to_keep_neighbors]
            
            G1.add_vertices(len(G0.vs))
            G1.vs["name"] = G0.vs["name"]
            G1.vs["sec_type"] = G0.vs["sec_type"]
            G1.add_edges(G1_edges)
            G1.es["distance"] = G1_distances
            ssValues = []
            for vs in G1.vs:
                ssValue = ssTable[vs["name"][3][1]]
                ssValues.append(ssValue)
            G1.vs["ss"] = ssValues
            
                    
            GC = G1.components()
            components = [C for C in GC]
            L_components_classes = [len(set([ssTable[G1.vs[node1]["name"][3][1]] for node1 in component])) for component in components]
            
            strand_output = 1
            all_patterns_strands = []
            for component in components:
                gbc1 = [ssTable[G1.vs[node1]["name"][3][1]] for node1 in component]
                if len(set(gbc1)) > 1:    
                    strand_output = 0
                    break
                                    
            
                all_patterns_strands.append(gbc1[0])
                                                                    
            if len(all_patterns_strands) != len(set(all_patterns_strands)):
                strand_output = 0

            strands_OUTPUTS.append(strand_output)
            #print(strands_OUTPUTS)
        

            ######### Belonging to same beta sheet ############
            # Graph of pseudo beta strands (gammas)
            # A gamma is a list of residues. Connection between gammas? To determine between inter-distances between gammas
            #[[ssTable[G1.vs[node1]["name"][3][1]].split(":")[0] for node1 in component] for component in components]
            list_gammas = []
            pseudo_strands = []
            for component in components:
                gamma_names = [G1.vs[node1]["name"][3][1] for node1 in component]
                pseudo_strand = [ssTable[G1.vs[node1]["name"][3][1]].split(":")[0] for node1 in component]
                pseudo_strands.append(pseudo_strand)
                list_gammas.append(gamma_names)
                

            GB = igraph.Graph()
            GB.add_vertices(len(list_gammas))
            GB.vs["residues"] = list_gammas
            GB.vs["pseudo_strand"] = pseudo_strands
            GB_edges = [] 
            GB_distances = []
            Y_gamma = []
            X_gamma = []
            
            if len(GB.vs) == 0: # no beta sheet in the protein
                return [], [], [], [], [], [], [], [], [], [], []


            SHEET_DEBUG = []
            sheet_debug_labels = []

            
            X_tmp = []
            for i1, gamma1 in enumerate(GB.vs):
                residues_1 = gamma1["residues"]
                distances_gamma1 = []
                for i2, gamma2 in enumerate(GB.vs):
                    distances_gamma12 = []
                    if i1 == i2:
                        distances_gamma12 = [1e+10 for x in range(25)]
                    if i1 != i2: 
                        residues_2 = gamma2["residues"]
                        for residue1 in residues_1:
                            tmp_distances_gamma12 = []
                            for residue2 in residues_2:
                                key12 = " ".join(sorted([str(residue1), str(residue2)]))
                                #distances_gamma12.append(hashTable_D[key12])
                                tmp_distances_gamma12.append(hashTable_D[key12])

                            distances_gamma12.append(min(tmp_distances_gamma12))

                    #dg12 = np.mean(distances_gamma12) # to change perhaps
                    dg12 = np.mean(sorted(distances_gamma12)[:2])
                    distances_gamma1.append(dg12)
                    if 1:
                        i1_sheet = GB.vs[i1]["pseudo_strand"][0]
                        i2_sheet = GB.vs[i2]["pseudo_strand"][0]
                        sheet_test = int(i1_sheet == i2_sheet)
                        SHEET_DEBUG.append((sheet_test, i1, i2, distances_gamma12))
                        if sheet_test:
                            sheet_debug_labels.append(str(i1_sheet) + " " + str(i2_sheet))
                            

                    #if len(distances_gamma12) == 1 and distances_gamma12[0] != 1e+10:
                    #   pdb.set_trace()
                    X_tmp.append(distances_gamma12) 
                    if 1: # threshold based
                        #threshold_GB = 3.11
                        #if dg12 <  threshold_GB:
                        if 1:
                            GB_edges.append((i1, i2))
                            GB_distances.append(distances_gamma12)                        



            GB.add_edges(GB_edges)
            GB.es["distance"] = GB_distances

            GB_debug = GB.copy()
       
            if 1:
                GB = GB_debug.copy()
                #threshold_GB = 3.11
                GB.delete_edges([es.index for es in GB.es.select(lambda x : np.mean(sorted(x["distance"])[:2]) >=  threshold_GB )])
                GBC = [component for component in GB.components()]
                GBC1 = [[GB.vs[x]["pseudo_strand"] for x in component] for component in GBC]

                sheet_output = 1
                #pdb.set_trace()
                all_patterns = []
                for gbc1 in GBC1:
                    if len(set(sum(gbc1, []))) > 1:
                        sheet_output = 0
                        #print("NO")
                        break

                    pattern_set = set([])
                    for pseudo_strand in gbc1:
                        pattern_set.update(pseudo_strand)

                    for ps in pattern_set:
                        all_patterns.append(ps)

                if len(all_patterns) != len(set(all_patterns)):
                    sheet_output = 0

                    
                #print(all_patterns)
                SHEETS_outputs.append(sheet_output)

                set_sheet_labels = sorted(set(sum([sum(x, []) for x in GBC1], [])))
                ground_truth_sheets = []
                predicted_label_sheets = []
                for i1, gbc1 in enumerate(GBC1):
                    component_i1 = GBC[i1]
                    residues_i1 = [GB.vs[ii1]["residues"] for ii1 in component_i1]
                    residues_i1 = sum(residues_i1, [])
                    ground_truth_sheets += [ssTable[ri1].split(":")[0] for ri1 in residues_i1]
                    predicted_label_sheets += [i1 for ri1 in residues_i1]
                    #for pseudo_strand in gbc1:
                    #    predicted_label_sheet = i1
                    #    predicted_label_sheets.append(i1)

                ground_truth_vocabulary = sorted(set(ground_truth_sheets)) 
                ground_truth_sheets = [ground_truth_vocabulary.index(gts) for gts in ground_truth_sheets]
                h_score = v_measure_score(ground_truth_sheets, predicted_label_sheets) 
                H_scores.append(h_score)

    return A, X, Y, NX, features_FOS, X_gamma, Y_gamma, strands_OUTPUTS, SHEETS_outputs, H_scores


def most_common(lst):
    return max(set(lst), key=lst.count)

import time
import sys
import threading



def loadGraphsFromDirectory(filedir, name_to_pattern, nb_neighbors, classification_type="helices", distance_based=True, distance_threshold=5, nb_features=4, shuffle=False, nmr_conformations=False, nb_proteins=20):
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
    XF = []
    YF = []
    strands_outputs = []
    sheets_outputs = []
    clustering_scores = []
    S_OUTPUTS = []
    counter_protein = 0
    filew = open("listPDB1.txt", "w")
    for subdir, _, _ in os.walk(filedir):
        if subdir != filedir:
            for _, _, files in os.walk(subdir):
                for i_protein, protein_filename in enumerate(files):
                        suffix = protein_filename.split("/")[-1]
                        if counter_protein == nb_proteins - 1:
                            return As, Xs, Ys, NXs, FOSs, XF, YF, strands_outputs, S_OUTPUTS, clustering_scores

                        if '.cif' in protein_filename and suffix not in pathologic_cases:
                            #data_filename = protein_filename.split("/")[-1]
                            #print(data_filename)
                            sys.stdout.write("\rProtein filename, number %i, %s" % (i_protein, protein_filename))

                            #sys.stdout.flush()
                            if nmr_conformations:
                                ss_filename = "/data/PDB/TALOS/SecStruct_TALOS/" + protein_filename.split(".cif")[0] + ".ss"
                            else:
                                ss_filename = "/data/PDB/cullpdb/SecStruct_dunbrack.3757/" + protein_filename.split(".cif")[0] + ".ss"
                            A, X, Y, NX, FOS, Xf, Yf, strand_OUTPUTS, S_outputs, clustering_score = loadGraphFromFile(subdir + "/"  + protein_filename, ss_filename ,name_to_pattern, nb_neighbors, nmr_conformations=nmr_conformations)
                            if len(A) > 0:
                                As.append(A)
                                Xs.append(X)
                                Ys.append(Y)
                                NXs.append(NX)
                                FOSs.append(FOS)
                                XF += Xf
                                YF += Yf
                                strands_outputs.append(strand_OUTPUTS)
                                #sheets_outputs.append(sheet_output)
                                S_OUTPUTS.append(S_outputs)
                                #homogeneity_scores.append(homogeneity_score)
                                clustering_scores.append(clustering_score)
                                filew.write(protein_filename.split('.cif')[0] + "\n")
                                counter_protein += 1
                            #except:
                            #    continue
                        
    filew.close()
    return As, Xs, Ys, NXs, FOSs, XF, YF, strands_outputs, S_OUTPUTS, clustering_scores




if __name__ == "__main__":
    import pdb, sys, json
    from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
    #filename = sys.argv[1]
    #ss_filename = sys.argv[2]
    
    
    nb_neighbors = 2
    #A, X, Y, NX, features_FOS, _, _, _, _ = loadGraphFromFile(filename, ss_filename, name_to_pattern, nb_neighbors)
    
    nmr_conformations = int(sys.argv[1])
    #eta = float(sys.argv[2])
    #nmr_conformations = True
    if nmr_conformations:
        filedir = "/data/PDB/TALOS/PDBs_talos/"
        name_to_pattern = {}

    else:
        name_to_pattern = json.load(open("/data/PDB/cullpdb/cullpdb_dict.json", "r")) 
        filedir = "/data/PDB/cullpdb/"
    
    #threshold_G1 = 1.2
    #threshold_GB = 1.7

    with open("./pathologic_cases.txt", "r") as f:
        pathologic_cases = f.read().split(", ")
    
    sheet_scores = []
    H_scores = []
    #for threshold_GB in np.linspace(1.1,3,20).tolist():
    if 1:
        _,_,_,_,_, X0, Y0, strands_outputs, S_OUTPUTS, homogeneity_scores = loadGraphsFromDirectory(filedir, name_to_pattern, nb_neighbors, classification_type="all",distance_based=True, distance_threshold=3.5, nb_features=2, shuffle=True, nmr_conformations=nmr_conformations, nb_proteins=3600)
        final_S_scores = []
        final_H_scores = []
        final_st_scores = []
        nb_test = len(S_OUTPUTS[0])
        for i in range(nb_test):
            final_S_scores.append(sum([x[i] for x in S_OUTPUTS])/len(S_OUTPUTS))
            final_H_scores.append(sum([x[i] for x in homogeneity_scores])/len(homogeneity_scores))
            final_st_scores.append(sum([x[i] for x in strands_outputs])/len(strands_outputs))


     

    pdb.set_trace()


    print("False positives")
    print([x[2] for x in X0 if x[0] == 0 and x[5] == 0])
    max_dim = max([len(x[1]) for x in X0])
    min_dim = min([len(x[1]) for x in X0])
    X = [x[1][:min_dim] for x in X0]
    #X = []
    #for x0 in X0:
    #    x = x0.copy()
    #    if len(x) < max_dim: # fill with zeros
    #        for i in range(max_dim - len(x0)):
    #            x.append(0.)
    #    X.append(x)

    # Subsampling
    index_negative = [i for i, y in enumerate(Y0) if y==0]
    index_positive = [i for i, y in enumerate(Y0) if y==1]
    
    X_neg = [X[i] for i in index_negative]
    X_pos = [X[i] for i in index_positive[:len(index_negative)]]
    X = X_neg + X_pos
    Y = [0 for i in range(len(index_negative))] + [1 for i in range(len(index_negative))]


    A = list(enumerate(X))
    random.shuffle(A)
    indices_shuffle, X = zip(*A)
    Y = [Y[index] for index in indices_shuffle]
    X_rank = [X0[index][0] for index in indices_shuffle]


    #import matplotlib.pyplot as plt 
    #plt.boxplot([X[i] for i in range(len(Y)) if Y[i] == 0], patch_artist=True, boxprops=dict(facecolor="red", color="red"))
    #plt.boxplot([X[i] for i in range(len(Y)) if Y[i] == 1], patch_artist=True, boxprops=dict(facecolor="blue", color="blue"))
    #plt.show()



    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
    clf = DecisionTreeClassifier(random_state=0)
    cls = svm.SVC(C=3) 

    pdb.set_trace()

    clf.fit(X[:1000], Y[:1000])
    cls.fit(X[:1000], Y[:1000])

    print("DIMENSION")
    #print(dimension_F)
    
    print("SCORE")
    print(clf.score(X[1000:], Y[1000:]))
    print(cls.score(X[1000:], Y[1000:]))

    
                                                    
    pdb.set_trace()










    pdb.set_trace()

