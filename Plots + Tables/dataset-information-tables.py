# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 21:32:07 2019

@author: Mathias
"""

import os
import numpy as np
import pandas as pd

os.chdir('C:/Users/mathi/Desktop/PiB/')
pd.set_option('display.max_columns',50)
pd.set_option('display.max_rows',20)

dataset_info = pd.read_excel('./Data/dataset_information.xlsx', sheet_name = 'dataset_information')
dataset_info = dataset_info.loc[list(dataset_info.Sample_id.drop_duplicates().index)]

#################################################
#
#   Category Investigation
#
#################################################

# filter dataset
df_cat = dataset_info[~dataset_info.Metastasis_site.isnull()]
df_cat = df_cat[df_cat['Metastasis_site'] != 'unknown']

# filter platforms with >100 Sample_labels
platforms = df_cat.groupby('Platform_id').count()['Sample_label']
platforms = platforms[platforms > 100].index
df_cat = df_cat[df_cat.Platform_id.isin(platforms)]

# create pivot table
df_cat = pd.pivot_table(df_cat, values='Sample_id', index=['Platform_id'], columns=['Sample_label'], aggfunc=len)
df_cat = df_cat.fillna(0)

# add column and row total
df_cat['Total'] = df_cat.sum(axis=1)
df_cat.append(pd.Series(df_cat.sum(),name='Total'))

# Print table in latex format
print(df_cat.sort_index(ascending=False).to_latex(float_format = "{:0.0f}".format))



#################################################
#
#   Primary Site Investigation
#
#################################################

# filter dataset
df_cat = dataset_info[~dataset_info.Metastasis_site.isnull()]
df_cat = df_cat[df_cat['Metastasis_site'] != 'unknown']
df_cat = df_cat[df_cat.Sample_label == 'Metastasis Tumor']

# filter platforms with >100 Sample_labels
platforms = df_cat.groupby('Platform_id').count()['Sample_label']
platforms = platforms[platforms > 100].index
df_cat = df_cat[df_cat.Platform_id.isin(platforms)]

# create pivot table
df_cat = pd.pivot_table(df_cat, values='Sample_id', index=['Platform_id'], columns=['Primary_site'], aggfunc=len)
df_cat = df_cat.fillna(0)

# add column and row total
df_cat['Total'] = df_cat.sum(axis=1)
df_cat.append(pd.Series(df_cat.sum(),name='Total'))

print(df_cat.sort_index(ascending=False).to_latex(float_format = "{:0.0f}".format))
