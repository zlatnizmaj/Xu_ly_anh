import pandas as pd

path_files_CSV = "../dataset_modified_Input/"

df = pd.read_csv(path_files_CSV + 'Input_dataset.csv', header=None,
                 names=['Close', 'Change', 'Stock_Volatility',
                        'Stock_Momentum', 'Index_Volatility',
                        'Index_Momentum', 'Sector_Momentum'])

print(df.head())

# Output dataset for input train SVM
SVM_input = df[['Close', 'Index_Momentum', 'Index_Volatility',
                'Sector_Momentum',
                'Stock_Momentum', 'Stock_Volatility']]

SVM_input.to_csv(path_files_CSV + 'SVM_input.csv', index=None, header=False)

# Output dataset for target SVM (label)
SVM_target = df['Change']
SVM_target.to_csv(path_files_CSV + 'SVM_target.csv', index=None, header=False)
