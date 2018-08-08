import numpy as np
import csv
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
# Prepare DJIA training dataset

# Reading DJIA index prices csv file
with open('GSPC.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    # Converting the csv file reader to a lists
    data_list = list(spamreader)

# Separating header from the data
header = data_list[0]
data_list = data_list[1:]
data_list = np.asarray(data_list)

# Selecting date and close value for each day
selected_data = data_list[:, [0, 4, 5]]
df = pd.DataFrame(data=selected_data[0:, 1:],
                  index=selected_data[0:, 0],
                  columns=['close', 'adj close'],
                  dtype='float64')
print(df.head())

# Basic Data analysis for DJIA dataset
# copy the data to the new dataframe df1 which is temporary dataframe
df1 = df

# idx contains all the data value between given date range
idx = pd.date_range('07-31-2008', '07-31-2018')

# Adding missing dates to the dataframe
df1.index = pd.DatetimeIndex(df1.index)
df1 = df1.reindex(idx, fill_value=np.NaN)
print(df1.head())
df1.to_csv('df1.csv')
# Reference for pandas interpolation http://pandas.pydata.org/pandas-docs/stable/missing_data.html

interpolated_df = df1.interpolate()
print(interpolated_df.count())  # gives 3656 count

# Removing extra date rows added in data for calculating interpolation
interpolated_df = interpolated_df[:]
print(interpolated_df.head())

# Save pandas frame in csv form
interpolated_df.to_csv('GSPC_interpolated_df_10_years.csv')

# Load dataset
df_stocks = pd.read_csv('GSPC_interpolated_df_10_years.csv')

# Convert adj close price into integer format
df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)

# selecting the prices and articles
df_stocks = df_stocks[['prices']]
df = df_stocks[['prices']].copy()

print(df_stocks.head())
# df_stocks.to_csv('dfstocks.csv')

# Split training and testing data
train_start_date = '2008-07-31'
train_end_date = '2016-07-31'
test_start_date = '2017-08-01'
test_end_date = '2018-07-31'
train = df.loc[train_start_date: train_end_date]
test = df.loc[test_start_date:test_end_date]

# Split prediction labels for training and testing dataset
y_train = pd.DataFrame(train['prices'])
y_test = pd.DataFrame(test['prices'])

# SVM, SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(y_train)
print(svr_rbf.feature_importances_ )
idx = pd.date_range(test_start_date, test_end_date)
predictions_df = pd.DataFrame(data=prediction[0:], index = idx, columns=['prices'])