# Code to get Data
# For 5 companies
# With a gap of 4 weeks each, keeping the end date const, changing start time

import datetime
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data

from pandas_datareader import data as pdr

import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)

# # download dataframe
# data = pdr.get_data_yahoo("SPY", start="2017-01-01", end="2017-04-30")
#
# # download Panel
# data = pdr.get_data_yahoo(["SPY", "IWM"], start="2017-01-01", end="2017-04-30")
#
# start_date = datetime.datetime(2017,1,2)
# stockstoPull = 'ONGC.NS',' 'SBI', 'TCS.NS', 'ONGC.BO', 'BHEL.BO'
data = pdr.get_data_yahoo("ONGC.NS", start='2017-04-23', end='2017-05-24')
print(data)
for stockTicker in stockstoPull:
    print("Pulling Data of " + stockTicker)
    for i in range(12): 
        start_date += datetime.timedelta(weeks=-4)
        print(i+1)
        df = data.Dat(stockTicker, 'yahoo', start_date, '2017-01-30')
        abc = '.\DataSets\\' + str(i+1) + 'month' + stockTicker.partition('.')[0] + '.csv'
        print(abc)
        df.to_csv(abc)

for stockTicker in stockstoPull:
    df = data.DataReader(stockTicker, 'yahoo', '2017-01-05', '2017-01-05')
    abc = '.\DataSets\\' + '00Output'+ '.csv'
    print(abc)
    df.to_csv(abc)

