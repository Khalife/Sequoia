# Sequoia: Distance based secondary structure assignment with node classification


This repository contains the implementation of a method adressing the assignment of secondary structure in proteins, solely using inter atomic distances (or a subset of these distances), without knowing the protein sequence information. A protein is modelled as a graph of its atoms or residues depending on the geometric scale considered. In the case of the graphs of residues, edge features are computed using a generalization of the standard dihedral angle on a set of consecutive atoms. This computation is based on a geometric relation which allows the reconstruction dihedral angles (modulo their sign) based on inter-atomic distances. Then, we formalize the problem of secondary structure assignement based on node classification. We make use of a message passing neural network as one approximate solution. We also evaluate the impact of noise on the attribution of secondary structures, as well as the case where only distances between C-alpha atoms are known, which is a more realistic scenario in the case of Nuclear Magnetic Resonance (NMR) measurements.  





## Dependencies

scipy, Biopython, pytorch, pytorch-geometric, scikit-learn


## Scripts



### 0 - 

In the following, the filedir, datasetA or datasetB directories contain a subdirectory containing the .cif files. Example: 

filedir  -> 00 -> 1ja2.cif

datasetA -> 00 -> 1na2.cif, 1ba3.cif, ...

### 1 - simple\_baseline\_display.py

Parses .cif of .pdb data files and prints results obtained with the First Order Statistics (FOS) method. 

```python
python simple\_display.py filedir
```

### 2 - sequoia\_dataload\_multibio.py

Module used to construct distance-based features of the protein using .cif or .pdb files.

### 3 - sequoia\_datadump\_multibio.py


Writes to .pkl files extracted features (using 2 nearest neighbors) for each protein in the subfolders of filedir (cf. 0. for filedir format). The conformation file allows to consider one conformation per protein. Warning: if set to False, several conformations in the file may be used which may be overlapping in space.

python sequoia\_datadump\_multibio.py filedir output\_filename nb\_neighbors conformation calpha\_mode dssp\_mode conformation\_file

[comment]: <> (Ex: python sequoia\_datadump\_multibio.py cullpdb/ parsed\_data\_gnn\_cullpdb.pkl 2 xray 0 1)
Ex: python sequoia\_datadump\_multibio.py filedir\_example/ test\_output.pkl 2 xray False True cullpdb\_dict.json 


### 4 - sequoia\_train\_model.py

Trains a GNN for secondary structure prediction, using features distance-based features in pkl\_files (training set : 75%). 
Saves model parameters in the model\_path\_output file.

python sequoia\_train\_model.py train\_filename classification\_type nb\_neighbors model\_path\_output 

[comment]: <> (Ex: python sequoia\_train\_model.py parsed\_data\_gnn\_cullpdb.pkl helices 2 sequoia\_model\_save.tch)
Ex: python sequoia\_train\_model.py test\_output.pkl helices 2 test\_model\_output.tch

### 5 - sequoia\_infer\_secondary\_structures.py

Loads model in .tch file and infer secondary structures after parsing a .pdb of .cif file.

python sequoia\_infer\_secondary\_structures.py input\_filename classification\_type model\_filename calpha\_mode dssp\_mode output\_filename (optional: conformation\_table)

[comment]: <> (Ex: python sequoia\_infer\_secondary\_structures.py 1M22.cif helices sequoia\_model.tch  1 0 sequoia\_preds.txt cullpdb\_dict.json)

Ex: python sequoia\_infer\_secondary\_structures.py filedir\_example/00/2W3G.cif helices test\_model\_output.tch 0 1 sequoia\_preds.txt cullpdb\_dict.json

### 6 - create\_pml\_file.py

Reads output file predictions of sequoia\_infer\_secondary\_structures and construct .pml file for visualization with Pymol.
Uses zero\_residues.py to renumber residues.

python create\_pml\_file.py predictions\_filename input\_filename output\_directory   

Ex: python create\_pml\_file.py sequoia\_preds.txt 1M22.cif .   



[comment]: <> (5 - clustering\_edge\_multi\_load\_multiBio.py)

[comment]: <> (python noisy\_clustering\_edge\_multi\_load\_multiBio.py dataset\_index noise\_level)

[comment]: <> (The dataset\_index variables is 0 for datasetA and 1 for datasetB.)

[comment]: <> (Ex: python noisy\_clustering\_edge\_multi\_load\_multiBio.py 0 0.05)

[comment]: <> (Testing distance-based criteria for beta-sheet clustering. Returns score on list of .cif files)




## Datasets 

The list of pdb files for our Dataset A (X-ray cristallography) and Dataset B (NMR conformations) are in  datasetA/list\_proteins\_datasetA.txt and datasetB/list\_proteins\_datasetB.txt respectively.

datasetA and datasetB directories also .tgz files containing annotations used for beta-sheet clustering.




Example for datasetA from a directory containing list\_proteins\_datasetA.txt: 

mkdir datasetA && cd datasetA

mkdir 00 && cd 00

wget -i ../../list\_proteins\_datasetA.txt --no-check-certificate



Finally, some examples of trained models are given in the directory examples\_models\_data.


