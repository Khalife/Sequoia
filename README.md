# Sequoia: Distance based secondary structure assignment with node classification


This repository contains the implementation of a method adressing the assignment of secondary structure in proteins, solely using inter atomic distances (or a subset of these distances), without knowing the protein sequence information. A protein is modelled as a graph of its atoms or residues depending on the geometric scale considered. In the case of the graphs of residues, edge features are computed using a generalization of the standard dihedral angle on a set of consecutive atoms. This computation is based on a geometric relation which allows the reconstruction dihedral angles (modulo their sign) based on inter-atomic distances. Then, we formalize the problem of secondary structure assignement based on node classification. We make use of a message passing neural network as one approximate solution. We also evaluate the impact of noise on the attribution of secondary structures, as well as the case where only distances between C-alpha atoms are known, which is a more realistic scenario in the case of Nuclear Magnetic Resonance (NMR) measurements.  





## Dependencies

scipy, Biopython, pytorch, pytorch-geometric, scikit-learn

Tested and validated with python 3.8 and:

pytorch.__version__: 1.11.0+cu102

pytorch\_geometric.__version__: 2.04

Bio.__version__ 1.78

sklearn.__version__ 0.23.2 

## Scripts



### 0 - 

In the following, the filedir, datasetA or datasetB directories contain a subdirectory containing the .cif files. Example: 

filedir  -> 00 -> 1ja2.cif

datasetA -> 00 -> 1na2.cif, 1ba3.cif, ...

### 1 - simple\_baseline\_display.py

Parses .cif of .pdb data files and prints results obtained with the First Order Statistics (FOS) method. 

```python
python simple_display.py filedir
```

### 2 - sequoia\_dataload\_multibio.py

Module used to construct distance-based features of the protein using .cif or .pdb files.

### 3 - sequoia\_datadump\_multibio.py


Writes to .pkl files extracted features (using 2 nearest neighbors) for each protein in the subfolders of filedir (cf. 0. for filedir format). The conformation file allows to consider one conformation per protein. Warning: if set to False, several conformations in the file may be used which may be overlapping in space.

```python
python sequoia_datadump_multibio.py filedir output_filename nb_neighbors conformation calpha_mode dssp_mode conformation_file
```

[comment]: <> (Ex: python sequoia_datadump_multibio.py cullpdb/ parsed_data_gnn_cullpdb.pkl 2 xray 0 1)
Example:
```python
python sequoia_datadump_multibio.py filedir_example/ test_output.pkl 2 xray False True cullpdb_dict.json 
```


### 4 - sequoia\_train\_model.py

Trains a GNN for secondary structure prediction, using features distance-based features in pkl\_files (training set : 75%). 
Saves model parameters in the model\_path\_output file.

```python
python sequoia_train_model.py train_filename classification_type nb_neighbors model_path_output 
```

[comment]: <> (Ex: python sequoia_train_model.py parsed_data_gnn_cullpdb.pkl helices 2 sequoia_model_save.tch)
Example: 
```python
python sequoia_train_model.py test_output.pkl helices 2 test_model_output.tch
```

### 5 - sequoia\_infer\_secondary\_structures.py

Loads model in .tch file and infer secondary structures after parsing a .pdb of .cif file.

```python
python sequoia_infer_secondary_structures.py input_filename classification_type model_filename calpha_mode dssp_mode output_filename (optional: conformation_table)
```
[comment]: <> (Ex: python sequoia_infer_secondary_structures.py 1M22.cif helices sequoia_model.tch  1 0 sequoia_preds.txt cullpdb_dict.json)

Example:
```python
python sequoia_infer_secondary_structures.py filedir_example/00/2W3G.cif helices test_model_output.tch 0 1 sequoia_preds.txt cullpdb_dict.json
```

### 6 - create\_pml\_file.py

Reads output file predictions of sequoia\_infer\_secondary\_structures and construct .pml file for visualization with Pymol.
Uses zero\_residues.py to renumber residues.

```python
python create_pml_file.py predictions_filename input_filename output_directory   
```

Example:
```python
python create_pml_file.py sequoia_preds.txt 1M22.cif .   
```



[comment]: <> (5 - clustering_edge_multi_load_multiBio.py)

[comment]: <> (python noisy_clustering_edge_multi_load_multiBio.py dataset_index noise_level)

[comment]: <> (The dataset_index variables is 0 for datasetA and 1 for datasetB.)

[comment]: <> (Ex: python noisy_clustering_edge_multi_load_multiBio.py 0 0.05)

[comment]: <> (Testing distance-based criteria for beta-sheet clustering. Returns score on list of .cif files)




## Datasets 

The list of pdb files for our Dataset A (X-ray cristallography) and Dataset B (NMR conformations) are in  datasetA/list_proteins_datasetA.txt and datasetB/list_proteins_datasetB.txt respectively.

datasetA and datasetB directories also .tgz files containing annotations used for beta-sheet clustering.




Example for datasetA from a directory containing list_proteins_datasetA.txt: 

mkdir datasetA && cd datasetA

mkdir 00 && cd 00

wget -i ../../list_proteins_datasetA.txt --no-check-certificate



Finally, some examples of trained models are given in the directory examples_models_data.


