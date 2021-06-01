def writePmlFile(proteinFilename, writeFilename, list_prediction_labels):
    set_labels = sorted(list(set(list_prediction_labels)))
    n_labels = len(set_labels)
    classpoints = []
    for label in set_labels:
        classpoints.append([i for i, x in enumerate(list_prediction_labels) if x == label])

    split_filename = ""
    if ".pdb" in proteinFilename:
        proteinFilename_split = proteinFilename.split(".pdb")[0]
    if ".cif" in proteinFilename:
        proteinFilename_split = proteinFilename.split(".cif")[0]

    str_init = """cmd.load("%s") 
    bg_color black
    hide all
    run zero_residues.py
    zero_residues %s, 0
    
    set cartoon_oval_length, 0.7
    set cartoon_oval_width, 0.7
    set cartoon_rect_length, 0.7
    set cartoon_rect_width, 0.7
    set cartoon_loop_radius, 0.3
    """ % (proteinFilename, proteinFilename_split)

    str_mid = ''
    for i, class_ in enumerate(classpoints):
        str_mid += 'select class%s, resi %s\n' % ( i, "+".join([str(x) for x in class_]) )

    str_end = ''
    colors = ["marine", "purpleblue", "br7"] 
    for i, class_ in enumerate(classpoints):
        str_end += 'set cartoon_color, %s, class%s\n' %(colors[i], i)
    
    str_end += 'show cartoon, all'

    str_write = str_init + str_mid + str_end

    with open(writeFilename, "w") as f:
        f.write(str_write)


if __name__ == "__main__":
    import sys
    predictionsFilename = sys.argv[1]
    proteinFilename = sys.argv[2]
    with open(predictionsFilename, "r") as f:
        preds = f.read().split()
        preds = [int(pred) for pred in preds]
    writePmlFile(proteinFilename, "viz_pymol_" + proteinFilename.split(".")[0] + ".pml", preds)
