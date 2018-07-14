import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import statsmodels.api as sm

import patsy

'''
Conveniently, statsmodels comes with built-in datasets, 
so we can load a time-series dataset straight into memory
dataset called "Atmospheric CO2 from Continuous Air Samples 
at Mauna Loa Observatory, Hawaii, U.S.A.,"
'''
data = sm.datasets.co2.load_pandas()
co2 = data.data

print(co2.head())
print(co2.dtypes)
print(co2.index)

y = co2['co2'].resample('MS').mean()
print(y.head(5))

print('\n',y['1990':])
print(y['1995-10-01':'1996-10-01'])

print(y.isnull().sum())

