# CPWT
A deep learning pipeline for predicting protein interface contacts.
![image](https://github.com/wanglong0228/CPWT/assets/71430099/222161de-f891-49e9-b772-4c3a3b24e684)
# Dataset
The employed datasets are DB3, DB4, DB5,and DB5.5 from the family of Docking Benchmark (DB) datasets. Each dataset is split into training and test sets. DB5.5 is the latest and largest dataset consisting of totally 271 complexes, which provides the bound state of two given proteins as well as the unbound state of each protein. The DB5 is the subset of the DB5.5 with 230 complexes, which is most widely used. The DB3 and DB4 are subsets of DB5, which are relatively small. Specially, the DB3 contains 119 complexes and DB4 has 171 complexes totally.

# Data Preprossing 
Considering the irregular structure of proteins, graphs are constructed to describe them by taking residues as nodes with their edges defined according to the spatial relationship. On all datasets, we extract the same
features and labels by following the work in [29]. Generally, the features are divided into two types, i.e. sequence features and structural features. Specially, the Position Specific Scoring
Matrix(PSSM) is extracted from amino acid sequence as sequence features. For structural features, we extract Relative Accessible Surface Area(rASA), residue depth, protrusion index, hydrophobicity and half sphere amino acid composition. Finally, we combine all the aforementioned features, which result in 70-dimension features for nodes. Edge features are composed of average atomic distance and the angle between two residues [29]. To define the label, a pair of amino acids are treated to be interactive if any two non-hydrogen atoms, each from one amino acid, are within 6A˚, which is also adoptedby.
