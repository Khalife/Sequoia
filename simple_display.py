import pdb
import numpy as np
import Bio
from Bio import PDB
from Bio.Seq import Seq

def transformContextPadded(X, Y, w=10):
    n = len(X)
    feature_length = 4*w # 2 angles and bidirectionnal
    X_C = []
    if w > n:
        w = n
    for i in range(n):
        index_start = max([0, i-w])
        index_end = min([n, i+w]) 
        if i < w:
            complete_vector = [[0.,0.] for j in range(w-i)]
            X_c = sum(complete_vector + X[index_start:index_end], [])
        if i > n - w:
            complete_vector = [[0.,0.] for j in range(w-(n-i))]
            X_c = sum(X[index_start:index_end] + complete_vector, [])
         
        X_C.append(X_c)

    Y_C = Y
    return X_C, Y_C


def transformContext(X, Y, w=10, keep_value=False):
    n = len(X)
    X_C = []
    if w > n:
        w = n
    for i in range(n):
        index_start = max([0, i-(w-1)])
        index_end = min([n, i+(w-1)]) 
        X_i = X[i]
        if keep_value:
            X_c = []
            X_c.append(X_i)
            X_c += X[index_start:index_end] 
        else:
            X_c = X[index_start:index_end]
        X_C.append(X_c)

    Y_C = Y
    return X_C, Y_C



from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import *

parser = MMCIFParser()
protein_filenames = []

import sys
prefix = sys.argv[1] 
import os


#   H 	Alpha helix (4-12)
#   B 	Isolated beta-bridge residue
#   E 	Strand
#   G 	3-10 helix
#   I 	Pi helix
#   T 	Turn
#   S 	Bend
#   - 	None
map_st = {"G": 0, "H": 1, "I": 2, "T": 3, "E": 4, "B": 5, "S": 6, "-": 7}


protein_name_val = "1w09"

def parseData(filedir):
    X_phi_psi = []
    X_omega = []
    Y_labels = []
    debug_Y = []
    debug_counter = 0
    for subdir, _, _ in os.walk(filedir):
        if subdir != filedir:
            for _, _, files in os.walk(subdir):
                for protein_filename in files:
                    data_filename = protein_filename.split("/")[-1]
                    print(data_filename)
                    if debug_counter > 0:
                        break
                    if protein_name_val + ".cif" in data_filename and ".gz" not in data_filename:
                        print(subdir + "/" + protein_filename)
                        try:
                            structure = parser.get_structure(protein_filename, subdir + "/" + protein_filename)
                            dssp=PDB.DSSP(structure[0], subdir + "/"  + protein_filename)
                        except:
                            print("Could not parse or no file named", subdir + "/" + protein_filename)
                            continue
   
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
                        if len(dssp.keys()) > 100:
                            debug_counter += 1

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
                        
    return X_phi_psi, Y_labels, debug_Y


def basicFeatures(x, y, keep_value = False):
    if keep_value:
        phis = [xx[0] for xx in x[1:]]
        psis = [xx[1] for xx in x[1:]]
        avg1 = np.mean(phis)
        std1 = np.std(phis)
        avg2 = np.mean(psis)
        std2 = np.std(psis)
        return [x[0][0], x[0][1], avg1, std1, avg2, std2], y
    else:
        phis = [xx[0] for xx in x]
        psis = [xx[1] for xx in x]
        avg1 = np.mean(phis)
        std1 = np.std(phis)
        avg2 = np.mean(psis)
        std2 = np.std(psis)
        return [avg1, std1, avg2, std2], y

def basicFeaturesTrigo(x, y, keep_value = False):
    if keep_value:
        phis_cos = [np.cos(xx[0]*np.pi/180) for xx in x[1:]]
        psis_cos = [np.cos(xx[1]*np.pi/180) for xx in x[1:]]
        avg1_cos = np.mean(phis_cos)
        std1_cos = np.std(phis_cos)
        avg2_cos = np.mean(psis_cos)
        std2_cos = np.std(psis_cos)

        phis_sin = [np.sin(xx[0]*np.pi/180) for xx in x[1:]]
        psis_sin = [np.sin(xx[1]*np.pi/180) for xx in x[1:]]
        avg1_sin = np.mean(phis_sin)
        std1_sin = np.std(phis_sin)
        avg2_sin = np.mean(psis_sin)
        std2_sin = np.std(psis_sin)
        return [x[0][0], x[0][1], avg1_cos, std1_cos, avg2_cos, std2_cos], [x[0][0], x[0][1], avg1_sin, std1_sin, avg2_sin, std2_sin], y

    else:
        phis_cos = [np.cos(xx[0]*np.pi/180) for xx in x]
        psis_cos = [np.cos(xx[1]*np.pi/180) for xx in x]
        avg1_cos = np.mean(phis_cos)
        std1_cos = np.std(phis_cos)
        avg2_cos = np.mean(psis_cos)
        std2_cos = np.std(psis_cos)

        phis_sin = [np.sin(xx[0]*np.pi/180) for xx in x]
        psis_sin = [np.sin(xx[1]*np.pi/180) for xx in x]
        avg1_sin = np.mean(phis_sin)
        std1_sin = np.std(phis_sin)
        avg2_sin = np.mean(psis_sin)
        std2_sin = np.std(psis_sin)

        return [avg1_cos, std1_cos, avg2_cos, std2_cos], [avg1_sin, std1_sin, avg2_sin, std2_sin],  y


def allDihedralSTD(x, y):
    assert(not keep_value)
    std = np.std([xx[0] for xx in x[1:]] + [xx[1] for xx in x[1:]])
    return std, y


def allDihedralTrigoSTD(x, y):
    assert(not keep_value)
    std_cos = np.std([np.cos(xx[0]*np.pi/180) for xx in x[1:]] + [np.cos(xx[1]*np.pi/180) for xx in x[1:]])
    std_sin = np.std([np.sin(xx[0]*np.pi/180) for xx in x[1:]] + [np.sin(xx[1]*np.pi/180) for xx in x[1:]]) 

    return std_cos, std_sin, y



from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import itertools, operator
from sklearn.metrics import f1_score


keep_value = True
X_phi_psi_0, Y_labels_0, debug_Y = parseData(prefix)

#for w in [3,5,10,20]:
for w in [2]:
    X_phi_psi = []
    X_phi_psi2 = []
    Y_labels = []
    X_all = []
    X_all2 = []
    for X_protein, Y_protein in zip(X_phi_psi_0, Y_labels_0):
        x_phi_psi_new, y_label_new = transformContext(X_protein, Y_protein, w=w, keep_value=keep_value)
        x_phi_psi, x_phi_psi2, y_label, x_all, x_all2 = [], [], [], [], []
        for x, y in zip(x_phi_psi_new, y_label_new):
            x_phi_psi_c, x_phi_psi_s, y_label_c = basicFeaturesTrigo(x, y, keep_value=keep_value)

            x_all_c, x_all_s, _ = allDihedralTrigoSTD(x, y)
            x_all.append(x_all_c)
            x_all2.append(x_all_s)
            x_phi_psi.append(x_phi_psi_c)
            x_phi_psi2.append(x_phi_psi_s)
            y_label.append(y_label_c)
        X_phi_psi += x_phi_psi
        X_phi_psi2 += x_phi_psi2
        Y_labels += y_label
        X_all += x_all
        X_all2 += x_all2

    n_samples = len(X_phi_psi)
    n_train = int(n_samples*0.75)
    X_train = X_phi_psi[:n_train]
    X_test = X_phi_psi[n_train:]
    Y_train = Y_labels[:n_train]
    Y_test = Y_labels[n_train:]
    ####################################
    # Simple display
    # x-axis: residue index
    # y-axis: standard deviation of dihedral angle
    # Point color: blue: alpha helix, red: beta strand, black: other

    if not keep_value:
        X_std_phi = []
        X_std_psi = []
        X_std_phi2 = []
        X_std_psi2 = []
        for x in X_phi_psi:
            X_std_phi.append(x[1])
            X_std_psi.append(x[3])
        for x in X_phi_psi2:
            X_std_phi2.append(x[1])
            X_std_psi2.append(x[3])

    #   H   Alpha helix (4-12)
    #   B   Isolated beta-bridge residue
    #   E   Strand
    #   G   3-10 helix
    #   I   Pi helix
    #   T   Turn
    #   S   Bend
    #   -   Coil

    type_map = {'H': 'alpha', 'B': 'b bridge', 'E': 'b strand', 'G': '3-10 hlx', 'I': 'pi helix', 'T': 'turn', 'S': 'bend', '-': 'coil'}
    reverse_map_st_tmp = inv_map = {v: k for k, v in map_st.items()} 
    reverse_map_st = {}
    for key in reverse_map_st_tmp.keys():
        new_key = '$\\mathdefault{%s}$' % key 
        reverse_map_st[new_key] = type_map[reverse_map_st_tmp[key]]

    reverse_map_st['$\\mathdefault{nan}$'] = 'nan'


    import matplotlib.pyplot as plt
    fig, ((ax_phi, ax_psi, ax_all, ax_debug), (ax_phi2, ax_psi2, ax_all2, ax_debug2))= plt.subplots(2, 4)

    ax_phi.set_title("COS Phi STD (w=%i) vs. residue type" %w)
    ax_phi.set_xlabel("Residue index")
    ax_phi.set_ylabel("Standard deviation phi in context of residue")
    residue_indices = [i for i in range(len(X_std_phi))] 
    scatter = ax_phi.scatter(residue_indices, X_std_phi, c=Y_labels)
    legend1 = ax_phi.legend(*scatter.legend_elements(), loc="upper right", title="Type") 
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    labels_residues = [reverse_map_st[label] for label in labels]                       
    legend2 = ax_phi.legend(handles, labels_residues, loc="upper right", title="Type")
    ax_phi.add_artist(legend2)

    ax_psi.set_title("COS Psi STD (w=%i) vs. residue type" %w)
    ax_psi.set_xlabel("Residue index")
    ax_psi.set_ylabel("Standard deviation psi in context of residue")
    residue_indices = [i for i in range(len(X_std_psi))] 
    scatter = ax_psi.scatter(residue_indices, X_std_psi, c=Y_labels)
    legend1 = ax_psi.legend(*scatter.legend_elements(), loc="upper right", title="Type") #["alpha helix", "beta strand", "other"], loc="upper right", title="Type")

    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    labels_residues = [reverse_map_st[label] for label in labels]
    legend2 = ax_psi.legend(handles, labels_residues, loc="upper right", title="Type") 
    ax_psi.add_artist(legend2)

    ax_all.set_title("COS Phi/Psi STD (w=%i) vs. residue type" %w)
    ax_all.set_xlabel("Residue index")
    ax_all.set_ylabel("Standard deviation phi/psi in context of residue")
    residue_indices = [i for i in range(len(X_all))] 
    scatter = ax_all.scatter(residue_indices, X_all, c=Y_labels)
    legend1 = ax_all.legend(*scatter.legend_elements(), loc="upper right", title="Type") #["alpha helix", "beta strand", "other"], loc="upper right", title="Type")

    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    labels_residues = [reverse_map_st[label] for label in labels]
    legend2 = ax_all.legend(handles, labels_residues, loc="upper right", title="Type")                                                            
    ax_all.add_artist(legend2)

    ax_debug.set_title("Validation")
    ax_debug.set_xlabel("Residue index")
    ax_debug.set_ylabel("Nothing")
    residue_indices = [i for i in range(len(X_all))] 
    scatter = ax_debug.scatter(residue_indices, [1 for i in range(len(X_all))], c=Y_labels)
    legend1 = ax_debug.legend(*scatter.legend_elements(), loc="upper right", title="Type") #["alpha helix", "beta strand", "other"], loc="upper right", title="Type")

    
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    labels_residues = [reverse_map_st[label] for label in labels]
    legend2 = ax_debug.legend(handles, labels_residues, loc="upper right", title="Type")
    ax_debug.add_artist(legend2)

    ax_phi2.set_title("SIN Phi STD (w=%i) vs. residue type" %w)
    ax_phi2.set_xlabel("Residue index")
    ax_phi2.set_ylabel("Standard deviation phi in context of residue")
    residue_indices = [i for i in range(len(X_std_phi))] 
    scatter = ax_phi2.scatter(residue_indices, X_std_phi2, c=Y_labels)
    legend1 = ax_phi2.legend(*scatter.legend_elements(), loc="upper right", title="Type") #["alpha helix", "beta strand", "other"], loc="upper right", title="Type")
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    labels_residues = [reverse_map_st[label] for label in labels]
    legend2 = ax_phi2.legend(handles, labels_residues, loc="upper right", title="Type")                                                                   #ax_psi2.add_artist(legend1)
    ax_phi2.add_artist(legend2)

    ax_psi2.set_title("SIN Psi STD (w=%i) vs. residue type" %w)
    ax_psi2.set_xlabel("Residue index")
    ax_psi2.set_ylabel("Standard deviation psi in context of residue")
    residue_indices = [i for i in range(len(X_std_psi))] 
    scatter = ax_psi2.scatter(residue_indices, X_std_psi2, c=Y_labels)
    legend1 = ax_psi2.legend(*scatter.legend_elements(), loc="upper right", title="Type") #["alpha helix", "beta strand", "other"], loc="upper right", title="Type")
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    labels_residues = [reverse_map_st[label] for label in labels]
    legend2 = ax_psi2.legend(handles, labels_residues, loc="upper right", title="Type")
    ax_psi2.add_artist(legend2)
  
    ax_all2.set_title("SIN Phi/Psi STD (w=%i) vs. residue type" %w)
    ax_all2.set_xlabel("Residue index")
    ax_all2.set_ylabel("Standard deviation phi/psi in context of residue")
    residue_indices = [i for i in range(len(X_all))] 
    scatter = ax_all2.scatter(residue_indices, X_all2, c=Y_labels)
    legend1 = ax_all2.legend(*scatter.legend_elements(), loc="upper right", title="Type") #["alpha helix", "beta strand", "other"], loc="upper right", title="Type")

    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    labels_residues = [reverse_map_st[label] for label in labels]
    legend2 = ax_all2.legend(handles, labels_residues, loc="upper right", title="Type")
    ax_all2.add_artist(legend2)

    ax_debug2.set_title("Validation")
    ax_debug2.set_xlabel("Residue index")
    ax_debug2.set_ylabel("Nothing")
    residue_indices = [i for i in range(len(X_all))] 
    scatter = ax_debug2.scatter(residue_indices, [1 for i in range(len(X_all))], c=Y_labels)
    legend1 = ax_debug2.legend(*scatter.legend_elements(), loc="upper right", title="Type") #["alpha helix", "beta strand", "other"], loc="upper right", title="Type")
    ax_debug2.add_artist(legend1)

    plt.suptitle("Protein %s" %protein_name_val, fontsize=14)

    plt.show()

    ### k-nn
    knn_scores = []
    k_neighbors_values = [k for k in range(10, min([200, int((n_samples-1)/2)]), 10)]
    for k_neighbors in k_neighbors_values:
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        knn.fit(X_train, Y_train)
        knn_score = knn.score(X_test, Y_test)
        knn_scores.append(knn_score)
    
    import matplotlib.pyplot as plt
    plt.title("From dihedral angles to residue type: k-nn score")
    plt.xlabel("k")
    plt.ylabel("F1 score")
    
    plt.plot(k_neighbors_values, knn_scores)
    plt.show()
    ##################################
    
