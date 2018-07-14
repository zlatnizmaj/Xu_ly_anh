import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import statsmodels.api as sm

'''
Conveniently, statsmodels comes with built-in datasets, 
so we can load a time-series dataset straight into memory
dataset called "Atmospheric CO2 from Continuous Air Samples 
at Mauna Loa Observatory, Hawaii, U.S.A.,"
'''
data = sm.datasets.co2.load_pandas()
co2 = data.data

print(co2.head())

