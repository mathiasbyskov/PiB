# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:29:08 2019

@author: Mathias
"""

import os
import numpy as np
import pandas as pd
os.chdir('C:/Users/mathi/Desktop/PiB/')
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',20)

# Training + Test samples
training_samples = pd.read_csv('./Data/Samples/GPL570/train_samples.csv', header=0, index_col=False)
test_samples = pd.read_csv('./Data/Samples/GPL570/test_samples.csv', header=0, index_col=False)

methods = ['SVM']
datasets = ['CV-EXT', 'PCA', 'TSNE', 'MDS', 'ISOMAP', 'LLE']

for method in methods:
    
    # Initialize empty datasets
    results_train = pd.DataFrame(columns = ['method', 'dataset', 'params', 'mean_test_score'])
    results_test = pd.DataFrame(columns = ['method', 'dataset', 'params', 'mean_test_score', 'validation_score'])

    for dataset in datasets:
        
        print("\nFitting + Testing with {} on {}\n".format(method, dataset))
        
        """
            Step 1: Initialize dataset
            
        """
        
        # Load dataset
        df = pd.read_csv('./Data/Reduced DataFrames/{}/{}.csv'.format(dataset, dataset), header = 0, index_col = False)
        
        # Training data
        df_training = df[df.Sample.isin(training_samples.Sample)]
        X_train = df_training.iloc[:,5:]
        y_train = df_training.iloc[:,2]
        
        # Test data
        df_test = df[df.Sample.isin(test_samples.Sample)]
        X_test = df_test.iloc[:,5:]
        y_test = df_test.iloc[:,2]
        
        
        """
            Step 2: Initialize model
        
        """
        if method == 'Logistic':
            from sklearn.linear_model import LogisticRegression
            grid={ 'C': np.logspace(-5, 2, 25), 'penalty': ['l1', 'l2']}
            model = LogisticRegression(max_iter = 1000, multi_class = 'ovr', solver = 'liblinear') 
        
        if method == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier
            grid = { 'max_depth': [3, 5, 7, 9, 11, 13, 15, 17, None], 'max_features': [5, 8, 10, 12, 14, 16] }
            model = RandomForestClassifier(n_estimators = 5000)
        
        if method == 'XGboost':
            from xgboost import XGBClassifier
            grid = {
                'gamma': [0, 0.5, 1],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.1, 0.3, 0.5]
            }
            model = XGBClassifier(n_estimators = 800, objective = 'multi:softmax', num_class = 9, tree_method = 'exact')
            
        if method == 'SVM':
            from sklearn.svm import SVC
            grid = { 'C': np.logspace(-2, 1, 10), 'kernel': ['linear', 'rbf', 'sigmoid']}
            model = SVC(max_iter = -1)
            
        """
            Step 3: Fit model & Predict test-samples
        
        """
        
        from sklearn.model_selection import GridSearchCV
            
        model_cv = GridSearchCV(model, grid, cv = 5, scoring = 'accuracy', n_jobs = -1, verbose = 2, return_train_score = True)
        model_cv.fit(X_train, y_train)
            
        # Training Results
        current_res = pd.DataFrame(model_cv.cv_results_)[['params', 'mean_test_score']]
        current_res['method'] = method
        current_res['dataset'] = dataset
        
        results_train = pd.concat([results_train, current_res])
        
        
        # Test results
        from sklearn.metrics import accuracy_score
        
        y_pred = model_cv.predict(X_test)
        val_score = accuracy_score(y_test, y_pred)
        val_df = pd.DataFrame([[method, dataset, model_cv.best_params_, model_cv.best_score_, val_score]], columns = ['method', 'dataset', 'params', 'mean_test_score', 'validation_score'])
        results_test = pd.concat([results_test, val_df])
        
        print("Done fitting: {}, Train-score: {}, Test-score: {}".format(dataset, model_cv.best_score_, val_score))
    
    
    results_train.to_csv('./Results/TrainCV/{}-TrainCV.csv'.format(method), header = True, index = False)
    results_test.to_csv('./Results/Validation/{}-validation.csv'.format(method), header = True, index = False)
