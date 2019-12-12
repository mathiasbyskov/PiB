# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:38:41 2019

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

# Locally Linear Embedding 2 Components
from sklearn.manifold import LocallyLinearEmbedding as LLE
n_components = 2
neighbors = [3,4,5,6,7,8,9,10,11,12,13,14]

for neighbor in neighbors: # Prodcues dataset for each neighbor
    embedding = LLE(n_neighbors=neighbor, n_components=n_components, eigen_solver='dense', reg = 0.001)
    columns = ["LLE_{}".format(j+1) for j in range(n_components)]
    
    X_transformed = pd.DataFrame(embedding.fit_transform(X), columns = columns)
    
    pc_df = pd.concat([y, X_transformed], axis = 1, sort = False)
    pc_df.to_csv('./Data/Reduced DataFrames/LLE/LLE-{}N.csv'.format(neighbor), header = True, index = False)
    print("Round Done: {}".format(neighbor))
    
    
    

# 100 Components
from sklearn.manifold import LocallyLinearEmbedding as LLE
n_components = 100
neighbors = 13


embedding = LLE(n_neighbors=neighbors, n_components=n_components, eigen_solver='dense', reg = 0.001)
columns = ["LLE_{}".format(j+1) for j in range(n_components)]
    
X_transformed = pd.DataFrame(embedding.fit_transform(X), columns = columns)
    
pc_df = pd.concat([y, X_transformed], axis = 1, sort = False)
pc_df.to_csv('./Data/Reduced DataFrames/LLE/LLE.csv', header = True, index = False)
