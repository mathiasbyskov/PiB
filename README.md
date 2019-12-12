# Projects in Bioinformatics (PiB)

This repository holds the files and scripts related to "Project in Bioinformatics". A 10 ECTS project done at the Bioinformatics Research Center at Aarhus University.

## Project Description

**Title:**      
Prediction and Investigation of Cancerous Metastasis Gene Expression Data Through Machine Learning Methods

**Description:**     
In this proejct prediction and inference was done on gene-expression samples derived from the Human Cancer Metastasis Database. Various dimensionality reducing methods and classification methods were performed.

**Dimensionality reduction methods performed:**      
PCA, Multidimensional Scaling, Isomap, Locally Linear Embedding and t-SNE.

**Classification methods applied:**     
Logistic regression, Random Forest, XGBoost and Support Vector Machines.


## Structure of scripts:

### Data
Contains all scripts related to data extraction, merging and splitting along with the excel-file from HCMDB, that contained meta-information in regard to all the samples from the database.

1. dataset_information.xlsx: The file with meta-information downloaded from the HCMDB website;  https://hcmdb.i-sanger.com/
2. dataset.py: Uses the wrangling-file to collect wanted samples.
3. train_test_split: Did the stratified train/validation-split (80/20)
4. wrangling.py: Contains a class with methods for downloading samples from the GEO database and collecting them into a tidy dataframe

### ML
Contains all the scripts related to the various machine learning techniques used in the project.

1. Classification: Contains one file, where the 4 different ML classification methods are applied to the loaded dataset.
2. Dimension Reduction: Contains a file for each of the dimensionality reduction methods used.
3. DecisionTree.py: Constructs a decision tree and outputs the tree-plot.

### Plots + Tables
Contains all the scripts for producing the plots and tables used in project-report. Also the plots are included.

1. Introduction, Results and Appendix: Contains the plots produced by the scripts below.
2. The scripts all produces different plots that are stored in the directories mentioned above.
