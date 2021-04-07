# Distance based secondary structure assignment with node classification

## Datasets 

The list of pdb files for our Dataset A (X-ray cristallography) and Dataset B (NMR conformations) are in  datasetA/list\_proteins\_datasetA.txt and datasetB/list\_proteins\_datasetB.txt respectively.

datasetA and datasetB directories also .tgz files containing annotations used for beta-sheet clustering.




Example for datasetA: 


cd datasetA

gunzip -k SecStruct_DatasetA.tgz && tar -xvf SecStruct_DatasetA.tar

mkdir 00 && cd 00

wget -i ../../list\_proteins\_datasetA.txt









## Scripts





0 - In the following, the filedir, datasetA or datasetB directories contain a subdirectory containing the .cif files. Example: 

filedir  -> 00 -> 1ja2.cif

datasetA -> 00 -> 1ca2.cif
               -> 1ba3.cif

1 - simple\_display.py

python simple\_display.py filedir

Parses .cif data files and prints results obtained with the First Order Statistics (FOS) method. 

2 - edge\_multi\_load\_multiBio.py

Module used to construct distance-based features of the protein.

3 - write\_gcn\_multiBio.py

python write\_gcn\_multiBio.py filedir prefix\_output\_filename nb\_neighbors

Ex: python write\_gcn\_multiBio.py ./datasetA datasetA\_pickle 2

Writes to .pkl files extracted features for each protein in the subfolders of filedir (cf. 0. for filedir format).

4 - noisy\_gcn\_multiBio.py

python gcn\_multiBio.py pkl\_file test\_filename classification\_type noise\_level

Ex: python noisy\_gcn\_multiBio.py datasetA\_pickle.pkl datasetB\_pickle.pkl helices 0.10 

Trains a GNN for secondary structure prediction, using features distance-based features in pkl\_files (training set : 75%). 
Returns test score on the test set (25%).

5 - clustering\_edge\_multi\_load\_multiBio.py

python noisy\_clustering\_edge\_multi\_load\_multiBio.py dataset\_index noise\_level 

The dataset\_index variables is 0 for datasetA and 1 for datasetB.

Ex: python noisy\_clustering\_edge\_multi\_load\_multiBio.py 0 0.05

Testing distance-based criteria for beta-sheet clustering. Returns score on list of .cif files



