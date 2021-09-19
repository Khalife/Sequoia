if __name__ == "__main__":
    import sequoia_dataload_multibio
    import pdb, sys, pickle, json
    
    filedir = sys.argv[1]
    name_output = sys.argv[2]
    nb_neighbors = int(sys.argv[3])
    nmr_conformations = sys.argv[4]
    calpha_mode = int(sys.argv[5])
    dssp_mode = int(sys.argv[6])
    if nmr_conformations == "xray":
        nmr_conformations = False
        conformation_dict = sys.argv[7]
        name_to_pattern = json.load(open(conformation_dict, "r")) 
    else:
        nmr_conformations = True
        name_to_pattern = {}
    assert(len(name_output) > 0)
    As, Xs, Ys, NXs, FOSs = sequoia_dataload_multibio.loadGraphsFromDirectory(filedir, name_to_pattern, classification_type="all", distance_based=True, \
                                                                                distance_threshold=3.5, nb_neighbors=nb_neighbors, shuffle=True,\
                                                                                nmr_conformations=nmr_conformations, calpha_mode=calpha_mode, dssp_mode=dssp_mode)
    with open(name_output, "wb") as f:
        pickle.dump([As, Xs, Ys, NXs, FOSs], f)
