# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:39:34 2019

@author: Mathias
"""

import os
import pandas as pd
os.chdir('C:/Users/mathi/Desktop/PiB/')


df = pd.read_csv('./Data/Samples/GPL96_GPL570/GPL96_GPL570.csv', header=0, index_col=False)
df = df.dropna(axis='columns')
X = df.iloc[:,5:]
y = df.iloc[:,0:5]

# Scale
from sklearn import preprocessing
X = preprocessing.scale(X)

# PCA
from sklearn import decomposition
n_components = 100

pca = decomposition.PCA(n_components=n_components)
pca.fit(X)

X = pca.transform(X)
columns = ["PC{}".format(j+1) for j in range(n_components)]
    

pc_df = pd.DataFrame(X, columns = columns)
pc_df = pd.concat([y, pc_df], axis = 1, sort = False)
pc_df.to_csv('./Data/Reduced DataFrames/PCA//PCA.csv', header = True, index = False)


# Explained variance
import numpy as np

pca.explained_variance_
pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_) # Used to variance-explained plot


#################################################################
#
#   Variance Explained Plot (PCA)
#
##################################################################

import seaborn as sns
import matplotlib.pyplot as plt

var_exp = np.cumsum(pca.explained_variance_ratio_)
PCs = [(i+1) for i in range(100)]

frame = {'var_exp': pd.Series(var_exp), 'PCs': PCs}
variance_df = pd.DataFrame(frame)
plot_df = variance_df.iloc[0:50,:]


# PLotting
sns.set(font_scale=1.7)
plt.figure(figsize=(20,10))
g = sns.pointplot(x = 'PCs', y = 'var_exp', data=plot_df, scale = 0.8)

plt.xlabel('PCs', fontsize = 24)
plt.ylabel('% Variance Explained Cumulative', fontsize = 24)
g.set_xticklabels(g.get_xticklabels(), rotation=-45)
plt.ylim(0,1)
plt.savefig('./Plots + Tables/1. Introduction/pca-variance-exp.png')