
import _pickle as cPickle
import numpy as np
import pandas as pd
import datetime
from sklearn import preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import operator
from pandas_datareader import  data
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import re
from dateutil import parser
from backtest import Strategy, Portfolio


def getStock(symbol, start, end):
    """
    Downloads Stock from Yahoo Finance.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    df = pd.io.data.get_data_yahoo(symbol, start, end)

    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + symbol
    df['Return_%s' % symbol] = df['AdjClose_%s' % symbol].pct_change()

    return df


def getStock(symbol, start, end):
    """
    Downloads Stock from Yahoo Finance.
    Computes daily Returns based on Adj Close.
    Returns pandas dataframe.
    """
    df = data.get_data_yahoo(symbol, start, end)

    df.columns.values[-1] = 'AdjClose'
    df.columns = df.columns + '_' + symbol
    df['Return_%s' % symbol] = df['AdjClose_%s' % symbol].pct_change()

    return df

getStock()