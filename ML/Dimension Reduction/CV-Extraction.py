# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 16:06:03 2019

@author: Mathias
"""

import os
import pandas as pd
os.chdir('C:/Users/mathi/Desktop/PiB/')

# Load total dataframe
df = pd.read_csv('./Data/Samples/GPL570/GPL570.csv', header=0, index_col=False)
df = df.dropna(axis='columns')
X = df.iloc[:,5:]
y = df.iloc[:,0:5]

# Extract features with highest Coefficient of Variance (CV)
features = (X.std() / X.mean() * 100).sort_values(ascending=False)[0:100].index
X = X[features]

df_con = pd.concat([y, X], axis = 1, sort = False)
df_con.to_csv('./Data/Reduced DataFrames/CV-EXT/CV-EXT.csv', header = True, index = False)