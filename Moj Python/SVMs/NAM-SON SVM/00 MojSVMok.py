import os
import glob
from csv import reader
from sklearn import svm
import numpy as np
import pandas as pd

files_CSV = "./dataset/Input_dataset.csv"
# files = os.listdir(path_files_CSV)
# files = glob.glob(path_files_CSV+'/*.csv')
# print(files)

# directory = os.path.join(path_files_CSV, "path")
# for root, dirs, files in os.walk(directory):
#     for file in files:
#         if file.endswith(".csv"):
#             f = open(file, 'r')
#             #  perform calculation
#             f.close()


# Read datasets from CSV input file
def Read_file(file_name):
    dataset = list()
    with open(file_name, 'r', newline='', encoding='utf-8') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# df = Read_file(files_CSV)
df = pd.read_csv(files_CSV, header=None, names=['Close', 'Change', 'Stock_Volatility',
                                                'Stock_Momentum', 'Index_Volatility',
                                                'Index_Momentum', 'Sector_Momentum'])
# df = pd.read_csv("data/cereal.csv", skiprows = 1) # drop first row from csv
# df[0][1] = 'CloseMoj'
print(df.head())
print(df[['Change', 'Index_Momentum']])  # error
print(df['Close'].dtypes)
print(df['Close'][0] + df['Close'][1])
# print(df.ix[:, 0])
# print(df.ix[1, :])

# print(df.loc[:, 'Close'])

SVM_target = df['Change']
SVM_target.to_csv('SVM_target', index=None, header=False)
SVM_input = df[['Close', 'Index_Momentum', 'Index_Volatility',
                          'Sector_Momentum',
                          'Stock_Momentum', 'Stock_Volatility']]
SVM_input.to_csv('SVM_input', index=None, header=False)

# SVM_input = df['Close']

# df = pd.DataFrame.from_records(df[1:], columns=df[0])
# print(df.describe())
# print(df.shape)
# print(df.head())

# svm_target = pd.DataFrame({'Change': df['Change']})
# print(svm_target.shape)
# print(svm_target.values)
# convert_arr = svm_target.values
# convert_arr = convert_arr.ravel()
# print(convert_arr)
# print(convert_arr.shape)

# df_copy = df.copy()
# print(df_copy.head())
# old = pd.DataFrame({'A' : [4,5], 'B' : [10,20], 'C' : [100,50], 'D' : [-30,-50]})
# print(old)
# using dataframe

#df = pd.DataFrame(df)
