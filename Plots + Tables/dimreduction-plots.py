# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 19:18:29 2019

@author: Mathias
"""

import os
import pandas as pd
os.chdir('C:/Users/mathi/Desktop/PiB/')
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',20)

def replace_primary_site(df, column, to_replace, replacement):    
    for string in enumerate(to_replace):
        df[column] = df[column].str.replace(string[1], replacement[string[0]])        
    return df

####################################################################
#
#   Dimension Reduction Plot!
#
####################################################################

# Load dataframes
CV_EXT = pd.read_csv('./Data/Reduced DataFrames/CV-EXT/CV-EXT.csv', header = 0, index_col = False)
PCA  = pd.read_csv('./Data/Reduced DataFrames/PCA/PCA.csv', header = 0, index_col = False)
MDS  = pd.read_csv('./Data/Reduced DataFrames/MDS/MDS_2COMP.csv', header = 0, index_col = False)
ISO  = pd.read_csv('./Data/Reduced DataFrames/ISOMAP/ISOMAP-9N.csv', header = 0, index_col = False)
LLE  = pd.read_csv('./Data/Reduced DataFrames/LLE/LLE-13N.csv', header = 0, index_col = False)
TSNE = pd.read_csv('./Data/Reduced DataFrames/TSNE/TSNE_LR10_PER10_2COMP.csv', header = 0, index_col = False)


# Slice datasets
CV_EXT = CV_EXT.iloc[:,0:7]
PCA = PCA.iloc[:,0:7]
MDS = MDS.iloc[:,0:7]
ISO = ISO.iloc[:,0:7]
LLE = LLE.iloc[:,0:7]
TSNE  = TSNE.iloc[:,0:7]

# Rename columns
columns = ['Sample', 'Cancer_type', 'Primary_site', 'Metastasis_site', 'Sample_label', 'PC1', 'PC2']
CV_EXT.columns = columns
PCA.columns = columns
MDS.columns = columns    
ISO.columns = columns    
LLE.columns = columns    
TSNE.columns = columns    

# Add Method to DFs
CV_EXT['Method'] = 'CV_EXT'
PCA['Method'] = 'PCA'
MDS['Method'] = 'MDS'
ISO['Method'] = 'ISO'
LLE['Method'] = 'LLE'
TSNE['Method'] = 'TSNE'

# Merge DFs


CV_EXT = CV_EXT[CV_EXT.PC2 < 100]
CV_EXT = CV_EXT[CV_EXT.PC1 < 10000]

merged_df = pd.concat([CV_EXT, PCA, MDS, ISO, LLE, TSNE])


# Replace all Primary_site value
to_replace = list(merged_df.Primary_site.unique())
replacement = [i.capitalize() for i in to_replace]
replacement[3] = 'Kidney'

merged_df = replace_primary_site(merged_df, 'Primary_site', to_replace, replacement)
merged_df.rename(columns={'Primary_site':'Primary Site'}, inplace = True)

# PLOT (FINALLY!)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.75)
g = sns.lmplot(x='PC2', y='PC1', data = merged_df, hue = 'Primary Site', 
           fit_reg = False, col='Method', col_wrap=3, sharex=False, sharey=False, 
           legend=True, x_jitter=0.01, y_jitter=0.01, palette = "muted")

plt.savefig('./Plots + Tables/1. Introduction/dimreduction.png')





####################################################################
#
#   t-SNE plot
#
#####################################################################

LR = [10, 40, 70]
PER = [10, 30, 50, 70]


tSNE1 = pd.read_csv("./Data/Reduced DataFrames/TSNE/TSNE_LR10_PER10_2COMP.csv", header = 0, index_col = False)
tSNE2 = pd.read_csv("./Data/Reduced DataFrames/TSNE/TSNE_LR10_PER30_2COMP.csv", header = 0, index_col = False)
tSNE3 = pd.read_csv("./Data/Reduced DataFrames/TSNE/TSNE_LR10_PER50_2COMP.csv", header = 0, index_col = False)
tSNE4 = pd.read_csv("./Data/Reduced DataFrames/TSNE/TSNE_LR10_PER70_2COMP.csv", header = 0, index_col = False)
tSNE5 = pd.read_csv("./Data/Reduced DataFrames/TSNE/TSNE_LR40_PER10_2COMP.csv", header = 0, index_col = False)
tSNE6 = pd.read_csv("./Data/Reduced DataFrames/TSNE/TSNE_LR40_PER30_2COMP.csv", header = 0, index_col = False)
tSNE7 = pd.read_csv("./Data/Reduced DataFrames/TSNE/TSNE_LR40_PER50_2COMP.csv", header = 0, index_col = False)
tSNE8 = pd.read_csv("./Data/Reduced DataFrames/TSNE/TSNE_LR40_PER70_2COMP.csv", header = 0, index_col = False)
tSNE9 = pd.read_csv("./Data/Reduced DataFrames/TSNE/TSNE_LR70_PER10_2COMP.csv", header = 0, index_col = False)
tSNE10 = pd.read_csv("./Data/Reduced DataFrames/TSNE/TSNE_LR70_PER30_2COMP.csv", header = 0, index_col = False)
tSNE11 = pd.read_csv("./Data/Reduced DataFrames/TSNE/TSNE_LR70_PER50_2COMP.csv", header = 0, index_col = False)
tSNE12 = pd.read_csv("./Data/Reduced DataFrames/TSNE/TSNE_LR70_PER70_2COMP.csv", header = 0, index_col = False)

params = []
for lr in LR:
    for per in PER:
        params.append('LR: {}, PER: {}'.format(lr, per))

datasets = [tSNE1,tSNE2,tSNE3,tSNE4,tSNE5,tSNE6,tSNE7,tSNE8,tSNE9,tSNE10,tSNE11,tSNE12]
for dataset in enumerate(datasets):
    dataset[1]['Method'] = params[dataset[0]]
    
merged_df = pd.concat([tSNE1,tSNE2,tSNE3,tSNE4,tSNE5,tSNE6,tSNE7,tSNE8,tSNE9,tSNE10,tSNE11,tSNE12])

to_replace = list(merged_df.Primary_site.unique())
replacement = [i.capitalize() for i in to_replace]
replacement[3] = 'Kidney'
merged_df = replace_primary_site(merged_df, 'Primary_site', to_replace, replacement)

merged_df.rename(columns={'Primary_site':'Primary Site', 'TSNE_1': 'PC1', 'TSNE_2':'PC2'}, inplace = True)


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.75)
g = sns.lmplot(x='PC2', y='PC1', data = merged_df, hue = 'Primary Site', 
           fit_reg = False, col='Method', col_wrap=3, sharex=False, sharey=False, 
           legend=True, palette = "muted")


plt.savefig('./Plots + Tables/4. Appendix/tsne.png')



####################################################################
#
#   Isomap Plot
#
#####################################################################


ISO6 = pd.read_csv("./Data/Reduced DataFrames/ISOMAP/ISOMAP-6N.csv", header = 0, index_col = False).iloc[:,0:7]
ISO7 = pd.read_csv("./Data/Reduced DataFrames/ISOMAP/ISOMAP-7N.csv", header = 0, index_col = False).iloc[:,0:7]
ISO8 = pd.read_csv("./Data/Reduced DataFrames/ISOMAP/ISOMAP-8N.csv", header = 0, index_col = False).iloc[:,0:7]
ISO9 = pd.read_csv("./Data/Reduced DataFrames/ISOMAP/ISOMAP-9N.csv", header = 0, index_col = False).iloc[:,0:7]
ISO10 = pd.read_csv("./Data/Reduced DataFrames/ISOMAP/ISOMAP-10N.csv", header = 0, index_col = False).iloc[:,0:7]
ISO11 = pd.read_csv("./Data/Reduced DataFrames/ISOMAP/ISOMAP-11N.csv", header = 0, index_col = False).iloc[:,0:7]
ISO12 = pd.read_csv("./Data/Reduced DataFrames/ISOMAP/ISOMAP-12N.csv", header = 0, index_col = False).iloc[:,0:7]
ISO13 = pd.read_csv("./Data/Reduced DataFrames/ISOMAP/ISOMAP-13N.csv", header = 0, index_col = False).iloc[:,0:7]
ISO14 = pd.read_csv("./Data/Reduced DataFrames/ISOMAP/ISOMAP-14N.csv", header = 0, index_col = False).iloc[:,0:7]
ISO15 = pd.read_csv("./Data/Reduced DataFrames/ISOMAP/ISOMAP-15N.csv", header = 0, index_col = False).iloc[:,0:7]
ISO16 = pd.read_csv("./Data/Reduced DataFrames/ISOMAP/ISOMAP-16N.csv", header = 0, index_col = False).iloc[:,0:7]
ISO17 = pd.read_csv("./Data/Reduced DataFrames/ISOMAP/ISOMAP-17N.csv", header = 0, index_col = False).iloc[:,0:7]


neighbors = []
for i in range(5, 17):
    neighbors.append('{} Neighbors'.format(i+1))
    
for dataset in enumerate([ISO6,ISO7,ISO8,ISO9,ISO10,ISO11,ISO12,ISO13,ISO14,ISO15,ISO16,ISO17]):
    dataset[1]['Method'] = neighbors[dataset[0]]

merged_df = pd.concat([ISO6,ISO7,ISO8,ISO9,ISO10,ISO11,ISO12,ISO13,ISO14,ISO15,ISO16,ISO17])

to_replace = list(merged_df.Primary_site.unique())
replacement = [i.capitalize() for i in to_replace]
replacement[3] = 'Kidney'
merged_df = replace_primary_site(merged_df, 'Primary_site', to_replace, replacement)

merged_df.rename(columns={'Primary_site':'Primary Site', 'ISOMAP_1': 'PC1', 'ISOMAP_2':'PC2'}, inplace = True)


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.75)
g = sns.lmplot(x='PC2', y='PC1', data = merged_df, hue = 'Primary Site', 
           fit_reg = False, col='Method', col_wrap=3, sharex=False, sharey=False, 
           legend=True, palette = "muted")


plt.savefig('./Plots + Tables/4. Appendix/isomap.png')


####################################################################
#
#   LLE Plot
#
#####################################################################


LLE3 = pd.read_csv("./Data/Reduced DataFrames/LLE/LLE-3N.csv", header = 0, index_col = False).iloc[:,0:7]
LLE4 = pd.read_csv("./Data/Reduced DataFrames/LLE/LLE-4N.csv", header = 0, index_col = False).iloc[:,0:7]
LLE5 = pd.read_csv("./Data/Reduced DataFrames/LLE/LLE-5N.csv", header = 0, index_col = False).iloc[:,0:7]
LLE6 = pd.read_csv("./Data/Reduced DataFrames/LLE/LLE-6N.csv", header = 0, index_col = False).iloc[:,0:7]
LLE7 = pd.read_csv("./Data/Reduced DataFrames/LLE/LLE-7N.csv", header = 0, index_col = False).iloc[:,0:7]
LLE8 = pd.read_csv("./Data/Reduced DataFrames/LLE/LLE-8N.csv", header = 0, index_col = False).iloc[:,0:7]
LLE9 = pd.read_csv("./Data/Reduced DataFrames/LLE/LLE-9N.csv", header = 0, index_col = False).iloc[:,0:7]
LLE10 = pd.read_csv("./Data/Reduced DataFrames/LLE/LLE-10N.csv", header = 0, index_col = False).iloc[:,0:7]
LLE11 = pd.read_csv("./Data/Reduced DataFrames/LLE/LLE-11N.csv", header = 0, index_col = False).iloc[:,0:7]
LLE12 = pd.read_csv("./Data/Reduced DataFrames/LLE/LLE-12N.csv", header = 0, index_col = False).iloc[:,0:7]
LLE13 = pd.read_csv("./Data/Reduced DataFrames/LLE/LLE-13N.csv", header = 0, index_col = False).iloc[:,0:7]
LLE14 = pd.read_csv("./Data/Reduced DataFrames/LLE/LLE-14N.csv", header = 0, index_col = False).iloc[:,0:7]


neighbors = []
for i in range(2, 14):
    neighbors.append('{} Neighbors'.format(i+1))
    
for dataset in enumerate([LLE3,LLE4,LLE5,LLE6,LLE7,LLE8,LLE9,LLE10,LLE11,LLE12,LLE13,LLE14]):
    dataset[1]['Method'] = neighbors[dataset[0]]

merged_df = pd.concat([LLE3,LLE4,LLE5,LLE6,LLE7,LLE8,LLE9,LLE10,LLE11,LLE12,LLE13,LLE14])

to_replace = list(merged_df.Primary_site.unique())
replacement = [i.capitalize() for i in to_replace]
replacement[3] = 'Kidney'
merged_df = replace_primary_site(merged_df, 'Primary_site', to_replace, replacement)

merged_df.rename(columns={'Primary_site':'Primary Site', 'LLE_1': 'PC1', 'LLE_2':'PC2'}, inplace = True)


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.75)
g = sns.lmplot(x='PC2', y='PC1', data = merged_df, hue = 'Primary Site', 
           fit_reg = False, col='Method', col_wrap=3, sharex=False, sharey=False, 
           legend=True, palette = "muted", x_jitter = 0.01, y_jitter=0.01)


plt.savefig('./Plots + Tables/4. Appendix/lle.png')



