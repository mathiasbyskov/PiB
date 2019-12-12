# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:41:43 2019

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

# ISOMAPPING 2 Components
from sklearn.manifold import Isomap
n_components = 2
neighbors = [6,7,8,9,10,11,12,13,14,15,16,17]

for neighbor in neighbors: # Prodcues dataset for each neighbor
    embedding = Isomap(n_neighbors=neighbor, n_components=n_components, eigen_solver='dense')
    columns = ["ISOMAP_{}".format(j+1) for j in range(n_components)]
    
    X_transformed = pd.DataFrame(embedding.fit_transform(X), columns = columns)
    
    pc_df = pd.concat([y, X_transformed], axis = 1, sort = False)
    pc_df.to_csv('./Data/Reduced DataFrames/ISOMAP/ISOMAP-{}N.csv'.format(neighbor), header = True, index = False)
    print("Round Done: {}".format(neighbor))
    
    
# 100 components
from sklearn.manifold import Isomap
n_components = 100
neighbors = 9


embedding = Isomap(n_neighbors=neighbors, n_components=n_components, eigen_solver='dense')
columns = ["ISOMAP_{}".format(j+1) for j in range(n_components)]
    
X_transformed = pd.DataFrame(embedding.fit_transform(X), columns = columns)
    
pc_df = pd.concat([y, X_transformed], axis = 1, sort = False)
pc_df.to_csv('./Data/Reduced DataFrames/ISOMAP/ISOMAP.csv', header = True, index = False)
