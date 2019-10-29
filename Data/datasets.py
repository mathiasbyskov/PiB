
import os
os.chdir('C:\\Users\\mathi\\Desktop\\PiB')

import pandas as pd
import wrangling
import Plots.plots


#### DOWNLOAD AND CREATE DATAFRAMES #####

datasets = ['GPL96', 'GPL570', 'GPL8432', 'GPL10379', 'GPL10558', 'GPL15659']
download = True

for data in datasets:

    # Filter dataframe
    dataset_info = pd.read_excel('dataset_information.xlsx', sheet_name = 'dataset_information')
    df_sample = dataset_info[(dataset_info.Platform_id == data) & (dataset_info.Sample_label == 'Metastasis Tumor') & (dataset_info.Metastasis_site != 'unknown')]
    df_sample = df_sample.rename(columns={'Sample_id': 'Sample'})
    df_sample = df_sample[['Sample', 'Cancer_type', 'Primary_site', 'Metastasis_site', 'Sample_label']]
    df_sample = df_sample.drop_duplicates()
    
    # Collect data into pd-dataframe from sample-list
    samples = set(df_sample.Sample.tolist())
    df_geneexp = wrangling.GSM_data(samples, download = download, filepath = str('./Samples/' + data + '/'), silent = True).df
    df_merged = pd.merge(df_sample, df_geneexp, on='Sample', how='inner')
    df_merged.to_csv(str('./Samples/' + data + '/' + data + '.csv'), header = True, index = False)
    



#### LOAD ALREADY EXISTING DATAFRAMES #####
    
GPL96 = pd.read_csv('./Data/Samples/GPL96/GPL96.csv', header = 0, index_col = False)
GPL570 = pd.read_csv('./Data/Samples/GPL570/GPL570.csv', header = 0, index_col = False)
GPL8432 = pd.read_csv('./Data/Samples/GPL8432/GPL8432.csv', header = 0, index_col = False)
GPL10379 = pd.read_csv('./Data/Samples/GPL10379/GPL10379.csv', header = 0, index_col = False)
GPL10558 = pd.read_csv('./Data/Samples/GPL10558/GPL10558.csv', header = 0, index_col = False)
GPL15659 = pd.read_csv('./Data/Samples/GPL15659/GPL15659.csv', header = 0, index_col = False)




# NO. SAMPLES!

    # GPL96: 167
    # GPL570: 192
    # GPL8432: 104
    # GPL10379: 207
    # GPL10558: 263
    # GPL15659: 145


# POSSIBLE MERGES!
    # GPL96 + GPL570: 359
    # (MAYBE: + GPL10379): 566
    # GPL8432 + 10558: 367
