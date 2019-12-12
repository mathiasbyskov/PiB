# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:12:19 2019

@author: Mathias
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir('C:/Users/mathi/Desktop/PiB/')
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',20)


################################################
#
#   Investigation of Missing Values in GPL's
#
################################################

GPL96 = pd.read_csv('./Data/Samples/GPL96/GPL96.csv', header = 0, index_col = False).iloc[:,5:]
GPL570 = pd.read_csv('./Data/Samples/GPL570/GPL570.csv', header = 0, index_col = False).iloc[:,5:]

GPL96 = GPL96.isnull().sum(axis=0) / 167
GPL570 = GPL570.isnull().sum(axis=0) / 167

# Plotting
sns.set(font_scale=1.8)
plt.figure(figsize=(20,10))
g2 = sns.distplot(GPL570, hist = True, kde = False, label = 'GPL570', hist_kws={"alpha": 0.75, 'color': 'royalblue'}, bins= 3)
g1 = sns.distplot(GPL96, hist = True, kde = False, label = 'GPL96', hist_kws={"alpha": 0.75, 'color':'coral'}, bins = 38)
plt.xlabel('% of Missing Values')
plt.ylabel('Frequency')
plt.legend(loc = 'upper right')
plt.xlim(0,1)
plt.savefig('./Plots + Tables/1. Introduction/missing-values-individual-dfs.png')


################################################
#
#   Investigation of Missing Values merged df
#
################################################


merged_df = pd.read_csv('./Data/Samples/GPL96_GPL570/GPL96_GPL570.csv', header = 0, index_col = False).iloc[:,5:]

missing_values = merged_df.isnull().sum(axis=0) / merged_df.shape[0]

sns.set(font_scale=1.8)
plt.figure(figsize=(20,10))
g1 = sns.distplot(missing_values, hist = True, kde = False, label = 'GPL96', bins = 8, hist_kws={"alpha": 0.9})
plt.xlabel('% of Missing Values')
plt.ylabel('Frequency')
plt.legend(loc = 'upper right')
plt.xlim(0,1)
plt.savefig('./Plots + Tables/1. Introduction/missing-values-merged.png')





################################################
#
#   Investigation of CV in GPL's
#
################################################

GPL96 = pd.read_csv('./Data/Samples/GPL96/GPL96.csv', header = 0, index_col = False).iloc[:,5:]
GPL570 = pd.read_csv('./Data/Samples/GPL570/GPL570.csv', header = 0, index_col = False).iloc[:,5:]

df_merged = pd.concat([GPL96, GPL570], sort = True)
df_merged = df_merged.dropna(axis='columns')

cols = list(df_merged.columns)

GPL96 = GPL96[cols]
GPL570 = GPL570[cols]


GPL96 = (GPL96.std() / GPL96.mean() * 100)
GPL570 = (GPL570.std() / GPL570.mean() * 100)
df_merged = (df_merged.std() / df_merged.mean() * 100)

# Plotting
sns.set(font_scale=1.8)
plt.figure(figsize=(20,10))
g1 = sns.distplot(GPL96, hist = True, kde = False, label = 'GPL96', hist_kws={"alpha": 0.75, 'color':'coral'})
g2 = sns.distplot(GPL570, hist = True, kde = False, label = 'GPL570', hist_kws={"alpha": 0.75, 'color': 'royalblue'})
plt.xlabel('Coefficient of Variance', fontsize = 28)
plt.ylabel('Frequency', fontsize = 28)
plt.legend(loc = 'upper right')
plt.xlim(0,500)
plt.savefig('./Plots + Tables/1. Introduction/CV-investigation.png')


######################################################
#
#   PCA Plot of GPL's
#
######################################################

GPL96 = pd.read_csv('./Data/Samples/GPL96/GPL96.csv', header = 0, index_col = False)
GPL570 = pd.read_csv('./Data/Samples/GPL570/GPL570.csv', header = 0, index_col = False)

GPL96['Platform'] = 'GPL96'
GPL570['Platform'] = 'GPL570'

df_merged = pd.concat([GPL96, GPL570], sort = True)
df_merged = df_merged.dropna(axis='columns')

to_replace = list(df_merged.Primary_site.unique())
replacement = [i.capitalize() for i in to_replace]
replacement[7] = 'Kidney'

def replace_primary_site(df, column, to_replace, replacement):    
    for string in enumerate(to_replace):
        df[column] = df[column].str.replace(string[1], replacement[string[0]])        
    return df

df_merged = replace_primary_site(df_merged, 'Primary_site', to_replace, replacement)


df_merged.rename(columns={'Primary_site':'Primary Site'}, inplace = True)

df_merged = df_merged[(df_merged['Primary Site'] == 'Breast') | (df_merged['Primary Site'] == 'Colorectum')]


# Scale
from sklearn import preprocessing
X = preprocessing.scale(df_merged.iloc[:,0:-6])

# PCA
from sklearn import decomposition
n_components = 2

pca = decomposition.PCA(n_components=n_components)
pca.fit(X)

X = pca.transform(X)
columns = ["PC{}".format(j+1) for j in range(n_components)]
    

pc_df = pd.DataFrame(X, columns = columns)
pc_df = pd.concat([df_merged.reset_index().iloc[:,-3], df_merged.reset_index().iloc[:,-4], pc_df], axis = 1, sort = False)

import seaborn as sns
import matplotlib.pyplot as plt

# PLotting
sns.set(font_scale=2.2)
plt.figure(figsize=(18,10))
g = sns.scatterplot(x = 'PC1', y = 'PC2', data=pc_df, hue = 'Primary Site', style = 'Platform', s = 110)
plt.xlabel('PC1', fontsize = 26)
plt.ylabel('PC2', fontsize = 26)
#plt.xlim(-95, -85)
#plt.ylim(-5,10)
plt.legend(loc = 'center right', bbox_to_anchor = (1, 0.8, 0, 0), ncol = 1, labelspacing = 1, markerscale = 2)
plt.savefig('./Plots + Tables/1. Introduction/PCA1.png')


# PLotting
sns.set(font_scale=2.2)
plt.figure(figsize=(18,10))
g = sns.scatterplot(x = 'PC1', y = 'PC2', data=pc_df, hue = 'Primary Site', style = 'Platform', s = 110)
plt.xlabel('PC1', fontsize = 26)
plt.ylabel('PC2', fontsize = 26)
plt.xlim(-95, -85)
plt.ylim(-5,10)
plt.legend(loc = 'center right', bbox_to_anchor = (1, 0.8, 0, 0), ncol = 1, labelspacing = 1, markerscale = 2)
plt.savefig('./Plots + Tables/1. Introduction/PCA1_zoomed.png')



####################################

#   Normalize before merging

####################################

GPL96 = pd.read_csv('./Data/Samples/GPL96/GPL96.csv', header = 0, index_col = False)
GPL570 = pd.read_csv('./Data/Samples/GPL570/GPL570.csv', header = 0, index_col = False)

GPL96['Platform'] = 'GPL96'
GPL570['Platform'] = 'GPL570'

df_merged = pd.concat([GPL96, GPL570], sort = True)
df_merged = df_merged.dropna(axis='columns')

cols = list(df_merged.columns)

GPL96 = GPL96[cols]
GPL570 = GPL570[cols]


# Scale
from sklearn import preprocessing
X96 = preprocessing.scale(GPL96.iloc[:,0:-6])
X570 = preprocessing.scale(GPL570.iloc[:,0:-6])

X_merged = pd.concat([pd.DataFrame(X96, columns = cols[:-6]), pd.DataFrame(X570, columns = cols[:-6])], sort = True)
X_merged = X_merged.reset_index()

X_merged = pd.concat([X_merged, df_merged.iloc[:,-6:].reset_index()], axis = 1, sort=True)
X_merged.drop('index', axis = 'columns', inplace = True)
X_merged = X_merged[(X_merged['Primary_site'] == 'breast') | (X_merged['Primary_site'] == 'colorectum')]


# PCA
from sklearn import decomposition
n_components = 2

pca = decomposition.PCA(n_components=n_components)
pca.fit(X_merged.iloc[:,:-6])

X = pca.transform(X_merged.iloc[:,:-6])
columns = ["PC{}".format(j+1) for j in range(n_components)]


pc_df = pd.DataFrame(X, columns = columns)

pc_df = pd.concat([pc_df, X_merged.iloc[:,-3].reset_index(drop=True), X_merged.iloc[:,-4].reset_index(drop=True)], axis = 1, sort = False)

to_replace = list(pc_df.Primary_site.unique())
replacement = [i.capitalize() for i in to_replace]

def replace_primary_site(df, column, to_replace, replacement):    
    for string in enumerate(to_replace):
        df[column] = df[column].str.replace(string[1], replacement[string[0]])        
    return df

pc_df = replace_primary_site(pc_df, 'Primary_site', to_replace, replacement)

pc_df.rename(columns={'Primary_site':'Primary Site'}, inplace = True)

import seaborn as sns
import matplotlib.pyplot as plt

# PLotting
sns.set(font_scale=1.8)
plt.figure(figsize=(20,10))
g = sns.scatterplot(x = 'PC1', y = 'PC2', data=pc_df, hue = 'Primary Site', style = 'Platform', s = 110)
plt.xlabel('PC1', fontsize = 28)
plt.ylabel('PC2', fontsize = 28)
#plt.xlim(-95, -85)
#plt.ylim(-5,10)
plt.legend(loc = 'center right', bbox_to_anchor = (1, 0.5, 0, 0), ncol = 1, labelspacing = 0.8, markerscale = 2)
plt.savefig('./Plots + Tables/4. Appendix/PCA_NORMALIZE.png')



