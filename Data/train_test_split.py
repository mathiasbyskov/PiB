# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 18:09:57 2019

@author: Mathias
"""

import os
import numpy as np
import pandas as pd
os.chdir('C:/Users/mathi/Desktop/PiB/')


df = pd.read_csv('./Data/Samples/GPL570/GPL570.csv', header=0, index_col=False)
df = df.dropna(axis='columns')
X = df.iloc[:,np.r_[0, 5:16662]] # All features including the 'Sample'
y = df.iloc[:,2]

# Train + Test-data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify = y, random_state = 100)

y_train.value_counts() # Count each Primary site in training-set
y_test.value_counts()  # Count each Primary site in test-set


# Save training + test samples
X_train.Sample.to_csv('./Data/Samples/GPL570/train_samples.csv', header = True, index = False)
X_test.Sample.to_csv('./Data/Samples/GPL570/test_samples.csv', header = True, index = False)


# Training samples: 153
# Test samples: 39

