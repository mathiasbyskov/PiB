# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:35:44 2019

@author: Mathias
"""

import os
import pandas as pd

os.chdir('C:/Users/mathi/Desktop/PiB/')
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',20)



#############################################################################
#
#   Import Data
#
#############################################################################

training_samples = pd.read_csv('./Data/Samples/GPL570/train_samples.csv', header=0, index_col=False)
test_samples = pd.read_csv('./Data/Samples/GPL570/test_samples.csv', header=0, index_col=False)
df = pd.read_csv('./Data/Reduced DataFrames/CV-EXT/CV-EXT.csv', header = 0, index_col = False)

# Training data
df_training = df[df.Sample.isin(training_samples.Sample)]
X_train = df_training.iloc[:,5:]
y_train = df_training.iloc[:,2]
        
# Test data
df_test = df[df.Sample.isin(test_samples.Sample)]
X_test = df_test.iloc[:,5:]
y_test = df_test.iloc[:,2]


#############################################################################
#
#   Classification Tree
#
#############################################################################

# Fit + Predict
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pydotplus

model = DecisionTreeClassifier(max_depth = 4)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score


train_score = accuracy_score(y_train, model.predict(X_train))
test_score = accuracy_score(y_test, model.predict(X_test))

print(train_score)
print(test_score)

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

# Visualize Tree
dot_data = tree.export_graphviz(model, out_file=None, filled=True, rounded=True,
                                feature_names=X_train.columns,  
                                class_names=y_train.unique())
graph = pydotplus.graph_from_dot_data(dot_data)  
graph.write_png('./Plots + Tables/3. Results/Tree.png')












