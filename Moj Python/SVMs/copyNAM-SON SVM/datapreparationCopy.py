import numpy as np
import csv
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
# Prepare DJIA training dataset

# Reading DJIA index prices csv file
with open('GSPCcopy.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    # Converting the csv file reader to a lists
    data_list = list(spamreader)

# Separating header from the data
header = data_list[0]
data_list = data_list[1:]

print('List data:\n', data_list)
for row in data_list:
    row[0] = row[0].split('-')
    row[0][:] = [''.join(row[0][:])]
    print(row)

# def concatenate_list_data(list):
#     result= ''
#     for element in list:
#         result += str(element)
#     return result
# print(concatenate_list_data([1, 5, 12, 2]))
#
# data_list = np.asarray(data_list)
# # Selecting date and close value for each day
# selected_data = data_list[:, [0, 5]]
# df = pd.DataFrame(data=selected_data[0:, 0:2],
#                   columns=['adj close'],)
#
# # print(selected_data[0:, 0:2])
# print(df.head())
#
# # Basic Data analysis for DJIA dataset
# # copy the data to the new dataframe df1 which is temporary dataframe
# df1 = df
#
# # idx contains all the data value between given date range
# idx = pd.date_range('07-31-2008', '08-20-2008')
#
# # Adding missing dates to the dataframe
# df1.index = pd.DatetimeIndex(df1.index)
# df1 = df1.reindex(idx, fill_value=np.NaN)
# print(df1.head())
# # df1.to_csv('df1.csv')
# # Reference for pandas interpolation http://pandas.pydata.org/pandas-docs/stable/missing_data.html
#
# interpolated_df = df1.interpolate()
# print(interpolated_df.count())  # gives 3656 count
#
# # Removing extra date rows added in data for calculating interpolation
# interpolated_df = interpolated_df[:]
# # print(interpolated_df.head())
#
# # Save pandas frame in csv form
# interpolated_df.to_csv('GSPC_interpolated_df_10_years.csv')
#
# # Load dataset
# df_stocks = pd.read_csv('GSPC_interpolated_df_10_years.csv')
#
# # Convert adj close price into integer format
# # df_stocks['adj close'].apply(np.int64)
# print(df_stocks[0])
# # print(pd.to_datetime(df_stocks[0], format='%Y-%m-%d', errors='raise', infer_datetime_format=False, exact=True))
# # selecting the prices and articles
# # df_stocks = df_stocks[['prices']]
# # df = df_stocks['adj close'].copy()
#
# print(df_stocks.head())
#
# # df_stocks.to_csv('dfstocks.csv')
#
# # Split training and testing data
# train_start_date = '2008-07-31'
# train_end_date = '2008-08-15'
# test_start_date = '2008-08-16'
# test_end_date = '2008-08-20'
# train = df.loc[train_start_date: train_end_date]
# test = df.loc[test_start_date:test_end_date]
#
# # Split prediction labels for training and testing dataset
# y_train = pd.DataFrame(train['adj close'])
# y_test = pd.DataFrame(test['adj close'])
#
# # SVM, SVR
# # svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# # svr_rbf.fit(y_train)
# # print(svr_rbf.feature_importances_ )
# # idx = pd.date_range(test_start_date, test_end_date)
# predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['prices'])