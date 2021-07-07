# Distance based secondary structure assignment with node classification


This repository contains the implementation of a method adressing the assignment of secondary structure in proteins, solely using inter atomic distances (or a subset of these distances), without knowing the protein sequence information. A protein is modelled as a graph of its atoms or residues depending on the geometric scale considered. In the case of the graphs of residues, edge features are computed using a generalization of the standard dihedral angle on a set of consecutive atoms. This computation is based on a geometric relation which allows the reconstruction dihedral angles (modulo their sign) based on inter-atomic distances. Then, we formalize the problem of secondary structure assignement based on node classification. We make use of a message passing neural network as one approximate solution. We also evaluate the impact of noise on the attribution of secondary structures, as well as the case where only distances between C-alpha atoms are known, which is a more realistic scenario in the case of Nuclear Magnetic Resonance (NMR) measurements.  





## Dependencies

scipy, Biopython, pytorch, pytorch-geometric, scikit-learn


## Scripts



0 - In the following, the filedir, datasetA or datasetB directories contain a subdirectory containing the .cif files. Example: 

filedir  -> 00 -> 1ja2.cif

datasetA -> 00 -> 1na2.cif, 1ba3.cif, ...

1 - simple\_baseline\_display.py

python simple\_display.py filedir

Parses .cif data files and prints results obtained with the First Order Statistics (FOS) method. 

2 - sequoia\_dataload\_multibio.py

Module used to construct distance-based features of the protein.

3 - sequoia\_datadump\_multibio.py

python sequoia\_datadump\_multibio.py filedir output\_filename nb\_neighbors conformation calpha\_mode dssp\_mode

Ex: python sequoia\_datadump\_multibio.py cullpdb/ parsed\_data\_gnn\_cullpdb.pkl 2 xray 0 1

Writes to .pkl files extracted features (using 2 nearest neighbors) for each protein in the subfolders of filedir (cf. 0. for filedir format). The conformation file allows to consider one conformation per protein. Warning: if set to False, several conformations in the file may be used which may be overlapping in space.

4 - sequoia\_train\_model.py

python sequoia\_train\_model.py train\_filename classification\_type model\_path\_output 

Ex: python sequoia\_train\_model.py parsed\_data\_gnn\_cullpdb.pkl helices 2 sequoia\_model\_save.tch

Trains a GNN for secondary structure prediction, using features distance-based features in pkl\_files (training set : 75%). 
Saves model parameters in the model\_path\_output file.

[comment]: <> (5 - clustering\_edge\_multi\_load\_multiBio.py)

[comment]: <> (python noisy\_clustering\_edge\_multi\_load\_multiBio.py dataset\_index noise\_level)

[comment]: <> (The dataset\_index variables is 0 for datasetA and 1 for datasetB.)

[comment]: <> (Ex: python noisy\_clustering\_edge\_multi\_load\_multiBio.py 0 0.05)

[comment]: <> (Testing distance-based criteria for beta-sheet clustering. Returns score on list of .cif files)




## Datasets 

The list of pdb files for our Dataset A (X-ray cristallography) and Dataset B (NMR conformations) are in  datasetA/list\_proteins\_datasetA.txt and datasetB/list\_proteins\_datasetB.txt respectively.

datasetA and datasetB directories also .tgz files containing annotations used for beta-sheet clustering.




Example for datasetA: 


cd datasetA

gunzip -k SecStruct_DatasetA.tgz && tar -xvf SecStruct_DatasetA.tar

mkdir 00 && cd 00

wget -i ../../list\_proteins\_datasetA.txt
