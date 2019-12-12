# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 17:12:38 2019

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


# tSNE 2 components
from sklearn.manifold import TSNE
n_components = 2
perplexities = [10, 30, 50, 70]
learning_rates = [10, 40, 70]

for per in perplexities:
    for lr in learning_rates:
        print("Fitting:  lr = {}, per = {}".format(lr,per))

        columns = ["TSNE_{}".format(j+1) for j in range(n_components)]
        X_transformed = pd.DataFrame(TSNE(n_components=n_components, method = 'exact', verbose = 2, n_iter = 800, learning_rate = lr, perplexity = per).fit_transform(X), columns = columns)

        pc_df = pd.concat([y, X_transformed], axis = 1, sort = False)
        pc_df.to_csv('./Data/Reduced DataFrames/TSNE/TSNE_LR{}_PER{}_2COMP.csv'.format(lr, per), header = True, index = False)



# 100 components
n_components = 100
lr = 10
per = 10
columns = ["TSNE_{}".format(j+1) for j in range(n_components)]
X_transformed = pd.DataFrame(TSNE(n_components=n_components, method = 'exact', verbose = 2, n_iter = 100000, learning_rate = 10, perplexity = 30).fit_transform(X), columns = columns)

pc_df = pd.concat([y, X_transformed], axis = 1, sort = False)
pc_df.to_csv('Data/Reduced DataFrames/TSNE/TSNE.csv', header = True, index = False)