import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []


def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)	# skipping column names
        for row in csvFileReader:
            # print(row)
            # print(row[0])
            print(row[0].split('-'))  # phan cot 1(0) theo dau '-'
            # dates.append(int(row[0].split('-')[0]))
            dates.append(row[0].split('-'))
            prices.append(float(row[1]))
    return


get_data('goog.csv')  # calling get_data method by passing the csv file to it
print("Dates- ", dates)
print("Prices- ", prices)
