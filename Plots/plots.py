
import os
os.chdir('C:\\Users\\mathi\\Desktop\\PiB\\Plots')
import pandas as pd


GPL96 = pd.read_csv('../Data/Samples/GPL96/GPL96.csv', header = 0, index_col = False)
GPL570 = pd.read_csv('../Data/Samples/GPL570/GPL570.csv', header = 0, index_col = False)
GPL8432 = pd.read_csv('../Data/Samples/GPL8432/GPL8432.csv', header = 0, index_col = False)
GPL10379 = pd.read_csv('../Data/Samples/GPL10379/GPL10379.csv', header = 0, index_col = False)
GPL10558 = pd.read_csv('../Data/Samples/GPL10558/GPL10558.csv', header = 0, index_col = False)
GPL15659 = pd.read_csv('../Data/Samples/GPL15659/GPL15659.csv', header = 0, index_col = False)


#### Functions to provide plots

def histogram_missing_values(path, 
                             grid = True, 
                             bins = 20, 
                             rwidth = 0.8, 
                             color = '#607c8e', 
                             alpha_y = 0, 
                             alpha_x = 0, 
                             y_label = "Genes"):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(path, header = 0, index_col = False)
    missing = 1 - (df.count() / len(df))
    missing.plot.hist(grid=grid, bins=bins, rwidth=rwidth, color=color)
    
    df_name = path.split('/')[-1]
    df_name = df_name.split('.')[0]
    
    plt.title(str(df_name + ' (Missing Values)'))
    plt.xlabel('% Missing Values')
    plt.ylabel('Freq.')
    plt.grid(axis='y', alpha=alpha_y)
    plt.grid(axis='x', alpha=alpha_x)
    plt.xlim(0, 1)
    plt.savefig(str(df_name + '-missingvalues.png'))
    plt.show()
    return None




# Plotting Missing Values
histogram_missing_values(str('../Data/Samples/' + 'GPL96' + '/' + 'GPL96' + '.csv'), bins = 15)
histogram_missing_values(str('../Data/Samples/' + 'GPL570' + '/' + 'GPL570' + '.csv'), bins = 2)
histogram_missing_values(str('../Data/Samples/' + 'GPL8432' + '/' + 'GPL8432' + '.csv'), bins = 15)
histogram_missing_values(str('../Data/Samples/' + 'GPL10379' + '/' + 'GPL10379' + '.csv'), bins = 10)
histogram_missing_values(str('../Data/Samples/' + 'GPL10558' + '/' + 'GPL10558' + '.csv'), bins = 10)
histogram_missing_values(str('../Data/Samples/' + 'GPL15659' + '/' + 'GPL15659' + '.csv'), bins = 14)

