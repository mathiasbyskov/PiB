# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:29:08 2019

@author: Mathias
"""

import os
import pandas as pd
os.chdir('C:/Users/mathi/Desktop/PiB/')


df = pd.read_csv('./Data/Samples/GPL570/GPL570.csv', header=0, index_col=False)
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

