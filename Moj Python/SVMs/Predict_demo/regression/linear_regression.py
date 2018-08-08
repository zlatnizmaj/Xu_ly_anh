# https://viblo.asia/p/mo-hinh-hoi-quy-ung-dung-trong-bai-toan-du-doan-gia-bat-dong-san-machine-learning-phan-2-xQMkJLrzGam
import os
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split


def getData():
    # Get home data from CSV file
    dataFile = None
    if os.path.exists('home_data.csv'):
        print("-- home_data.csv found locally")
        dataFile = pd.read_csv('home_data.csv', skipfooter=1, engine='python')

    return dataFile


def linearRegressionModel(X_train, Y_train, X_test, Y_test):
    linear = linear_model.LinearRegression()
    # Training process
    linear.fit(X_train, Y_train)
    # Evaluating the model
    score_trained = linear.score(X_test, Y_test)

    return score_trained


def lassoRegressionModel(X_train, Y_train, X_test, Y_test):
    lasso_linear = linear_model.Lasso(alpha=1.0)
    # Training process
    lasso_linear.fit(X_train, Y_train)
    # Evaluating the model
    score_trained = lasso_linear.score(X_test, Y_test)

    return score_trained


if __name__ == "__main__":
    data = getData()
    df = pd.DataFrame(data)
    print(df.head())
    if data is not None:
        # Selection few attributes
        attributes = list(
            [
                'num_bed',
                'year_built',
                'num_room',
                'num_bath',
                'living_area',
            ]
        )
        # Vector price of house
        Y = data['askprice']
        # print np.array(Y)
        # Vector attributes of house
        X = data[attributes]
        # Split data to training test and testing test
        X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2)
        # Linear Regression Model
        linearScore = linearRegressionModel(X_train, Y_train, X_test, Y_test)
        print ('Linear Score = ' , linearScore)
        # LASSO Regression Model
        lassoScore = lassoRegressionModel(X_train, Y_train, X_test, Y_test)
        print ('Lasso Score = ', lassoScore)