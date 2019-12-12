# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:15:41 2019

@author: Mathias
"""

import os
import pandas as pd
os.chdir('C:/Users/mathi/Desktop/PiB/')
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',20)

import seaborn as sns
import matplotlib.pyplot as plt

###############################################################
#
#   TrainCV Plot
#
###############################################################


logistic = pd.read_csv('./Results/TrainCV/logistic-TrainCV.csv', header = 0, index_col = False)
randomforest = pd.read_csv('./Results/TrainCV/randomForest-TrainCV.csv', header = 0, index_col = False)
xgboost = pd.read_csv('./Results/TrainCV/XGboost-TrainCV.csv', header = 0, index_col = False)
svm = pd.read_csv('./Results/TrainCV/SVM-TrainCV.csv', header = 0, index_col = False)

merged_df = pd.concat([logistic, randomforest, xgboost, svm])
merged_df['method'] = merged_df.method.str.replace('logistic', 'Logistic')
merged_df['method'] = merged_df.method.str.replace('RandomForest', 'RF')

# Strip chart
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
g = sns.stripplot(x='dataset', y='mean_test_score', data = merged_df, hue = 'method', 
                  palette = "muted", jitter = 0.25, alpha = 0.7, linewidth=1)

plt.ylabel('Train CV Accuracy', fontsize = 22)
plt.xlabel('Dataset', fontsize = 22)
plt.ylim(0,1)
plt.legend(loc='center left', bbox_to_anchor = (1, 0.5, 0, 0))
plt.savefig('./Plots + Tables/3. Results/traincv-strip.png')

# Boxplot
sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
g = sns.boxplot(x='dataset', y='mean_test_score', data = merged_df, hue = 'method', palette = "muted", 
                width = 0.5, linewidth = 1, fliersize=3)

plt.ylabel('Train CV Accuracy', fontsize = 22)
plt.xlabel('Dataset', fontsize = 22)
plt.ylim(0,1)
plt.legend(loc='center left', bbox_to_anchor = (1, 0.5, 0, 0), labelspacing = 1)
plt.savefig('./Plots + Tables/3. Results/traincv-box.png')


###############################################################
#
#   Train/Validation Plot
#
###############################################################

logistic = pd.read_csv('./Results/Validation/logistic-validation.csv', header = 0, index_col = False)
randomforest = pd.read_csv('./Results/Validation/randomForest-validation.csv', header = 0, index_col = False)
svm = pd.read_csv('./Results/Validation/SVM-validation.csv', header = 0, index_col = False)
xgboost = pd.read_csv('./Results/Validation/XGboost-validation.csv', header = 0, index_col = False)

merged_df = pd.concat([logistic, randomforest, xgboost, svm])
merged_df.columns = ['Method', 'Dataset', 'params', 'mean_test_score', 'validation_score']
merged_df['Method'] = merged_df.Method.str.replace('logistic', 'Logistic')
merged_df['Method'] = merged_df.Method.str.replace('RandomForest', 'RF')

sns.set(font_scale=1.5)
plt.figure(figsize=(20,10))
g = sns.scatterplot(x='mean_test_score', y='validation_score', data = merged_df, hue = 'Dataset', 
                  palette = "muted", style = 'Method', alpha = 1, linewidth = 1, s = 100)
plt.ylabel('Validation Accuracy', fontsize = 22)
plt.xlabel('Train CV Accuracy', fontsize = 22)
plt.plot([0, 1], [0, 1], linewidth=1, color="black")
plt.ylim(0.6,1.05)
plt.xlim(0.6,1.05)
plt.legend(loc = 'center right', bbox_to_anchor = (1.13, 0.5, 0, 0), ncol = 1, labelspacing = 0.7, markerscale = 1.5)

plt.savefig('./Plots + Tables/3. Results/train-val-scatter.png')