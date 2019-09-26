
import os
os.chdir('C:\\Users\\mathi\\Desktop\\PiB')

import pandas as pd
import wrangling
import plots

# Filter dataframe
dataset_info = pd.read_excel('Data/dataset_information.xlsx', sheet_name = 'dataset_information')
df_sample = dataset_info[(dataset_info.Platform_id == 'GPL15659') & (dataset_info.Sample_label == 'Metastasis Tumor') & (dataset_info.Metastasis_site != 'unknown')]
df_sample = df_sample.rename(columns={'Sample_id': 'Sample'})
df_sample = df_sample[['Sample', 'Cancer_type', 'Primary_site', 'Metastasis_site', 'Sample_label']]
df_sample = df_sample.drop_duplicates()

# Collect data into pd-dataframe from sample-list
samples = set(df_sample.Sample.tolist())
df_geneexp = wrangling.GSM_data(samples, download = True, filepath = './Data/Samples/GPL15659/', silent = True).df
df_geneexp.to_csv('./Data/Samples/GPL15659/GPL15659.csv', header = True, index = False)

# Merge sample-info with gene-info
df_merged = pd.merge(df_sample, df_geneexp, on='Sample', how='inner')

# Plot simple plots






# NO. SAMPLES
# GPL96: 167
# GPL570: 192
# GPL8432: 104
# GPL10379: 207
# GPL10558: 263
# GPL15659: 145


# POSSIBLE MERGES?
# GPL96 + GPL570: 359
# (MAYBE: + GPL10379): 566
# GPL8432 + 10558: 367
