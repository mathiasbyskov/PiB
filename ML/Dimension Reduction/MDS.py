# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:16:00 2019

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


# MDS 100 COMP
from sklearn.manifold import MDS
n_components = 100

embedding = MDS(n_components=n_components, verbose = 2, max_iter=100000, eps = 1e-6, n_init = 5)
columns = ["MDS_{}".format(j+1) for j in range(n_components)]

X_transformed = pd.DataFrame(embedding.fit_transform(X), columns = columns)
pc_df = pd.concat([y, X_transformed], axis = 1, sort = False)
pc_df.to_csv('./Data/Reduced DataFrames/MDS/MDS.csv', header = True, index = False)


# MDS 2 COMP
from sklearn.manifold import MDS
n_components = 2

embedding = MDS(n_components=n_components, verbose = 2, max_iter = 10000)
columns = ["MDS_{}".format(j+1) for j in range(n_components)]

X_transformed = pd.DataFrame(embedding.fit_transform(X), columns = columns)
pc_df = pd.concat([y, X_transformed], axis = 1, sort = False)
pc_df.to_csv('./Data/Reduced DataFrames/MDS/MDS_2COMP.csv', header = True, index = False)
