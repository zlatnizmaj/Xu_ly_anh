import pandas as pd

file1 = "../dataset_modified_Input/Apple_modified.csv"
file2 = "../dataset_modified_Input/Amazon_modified.csv"
file3 = "../dataset_modified_Input/Intel_modified.csv"
file4 = "../dataset_modified_Input/Microsoft_modified.csv"
file5 = "../dataset_modified_Input/Google_modified.csv"

fileIndex = "../dataset_modified_Input/NASDAQ_modified.csv"

xl1 = pd.read_csv(file1)  # xl1: Apple stock
date_1 = xl1['Date']
Close_1 = xl1['Close']
Change_1 = xl1['Change']
Momentum_1 = xl1['Momentum']

xl2 = pd.read_csv(file2)  # xl2: Amazon stock
date_2 = xl2['Date']
Close_2 = xl2['Close']
Change_2 = xl2['Change']
Momentum_2 = xl2['Momentum']

xl3 = pd.read_csv(file3)  # xl3: Intel stock
date_3 = xl3['Date']
Close_3 = xl3['Close']
Change_3 = xl3['Change']
Momentum_3 = xl3['Momentum']

xl4 = pd.read_csv(file4)  # xl4: Microsoft stock
date_4 = xl4['Date']
Close_4 = xl4['Close']
Change_4 = xl4['Change']
Momentum_4 = xl4['Momentum']


xl5 = pd.read_csv(file5)  # xl5: Google stock
date_5 = xl5['Date']
Close_5 = xl5['Close']
Change_5 = xl5['Change']
Momentum_5 = xl5['Momentum']

xlIndex = pd.read_csv(fileIndex)  # xlIndex: NASDAQ index
date_Index = xlIndex['Date']
Close_Index = xlIndex['Close']
Change_Index = xlIndex['Change']
Momentum_Index = xlIndex['Momentum']


numberOfDate = 5

Close, Change, Stock_Volatility, Stock_Momentum, Index_Volatility, Index_Momentum, Sector_Momentum = [], [], [], [], [], [], []

for i in range(5, 1261):

    # file1 stock process

    file1_close = (Change_1[i]+Change_1[i-1]+Change_1[i-2]+Change_1[i-3]+Change_1[i-4])/5
    file1_Stock_Momentum = (Momentum_1[i] + Momentum_1[i - 1] + Momentum_1[i - 2] + Momentum_1[i - 3] + Momentum_1[i - 4]) / numberOfDate

    # file2 stock process
    file2_close = (Change_2[i]+Change_2[i-1]+Change_2[i-2]+Change_2[i-3]+Change_2[i-4])/5
    file2_Stock_Momentum = (Momentum_2[i] + Momentum_2[i - 1] + Momentum_2[i - 2] + Momentum_2[i - 3] + Momentum_2[i - 4]) / numberOfDate

    # file3 stock process
    file3_close = (Change_3[i]+Change_3[i-1]+Change_3[i-2]+Change_3[i-3]+Change_3[i-4])/5
    file3_Stock_Momentum = (Momentum_3[i] + Momentum_3[i - 1] + Momentum_3[i - 2] + Momentum_3[i - 3] + Momentum_3[i - 4]) / numberOfDate

    # file4 stock process
    file4_close = (Change_4[i]+Change_4[i-1]+Change_4[i-2]+Change_4[i-3]+Change_4[i-4])/5
    file4_Stock_Momentum = (Momentum_4[i] + Momentum_4[i - 1] + Momentum_4[i - 2] + Momentum_4[i - 3] + Momentum_4[i - 4]) / numberOfDate

    # file5 stock process
    file5_close = (Change_5[i]+Change_5[i-1]+Change_5[i-2]+Change_5[i-3]+Change_5[i-4])/5
    file5_Stock_Momentum = (Momentum_5[i] + Momentum_5[i - 1] + Momentum_5[i - 2] + Momentum_5[i - 3] + Momentum_5[i - 4]) / numberOfDate

    # fileIndex stock process
    fileIndex_close = (Change_Index[i]+Change_Index[i-1]+Change_Index[i-2]+Change_Index[i-3]+Change_Index[i-4])/5
    fileIndex_Stock_Momentum = (Momentum_Index[i] + Momentum_Index[i - 1] + Momentum_Index[i - 2] + Momentum_Index[i - 3] + Momentum_Index[i - 4]) / numberOfDate

    # sector momentum for given day
    Sector_momentum = (file1_Stock_Momentum + file2_Stock_Momentum + file3_Stock_Momentum + file4_Stock_Momentum
                       + file5_Stock_Momentum)/5

    Close.append(Close_1[i])
    Change.append(Momentum_1[i])  # label predicted, momentum tai ngay thu i
    Stock_Volatility.append(file1_close)
    Stock_Momentum.append(file1_Stock_Momentum)
    Index_Volatility.append(fileIndex_close)
    Index_Momentum.append(fileIndex_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_2[i])
    Change.append(Momentum_2[i])
    Stock_Volatility.append(file2_close)
    Stock_Momentum.append(file2_Stock_Momentum)
    Index_Volatility.append(fileIndex_close)
    Index_Momentum.append(fileIndex_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_3[i])
    Change.append(Momentum_3[i])
    Stock_Volatility.append(file3_close)
    Stock_Momentum.append(file3_Stock_Momentum)
    Index_Volatility.append(fileIndex_close)
    Index_Momentum.append(fileIndex_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_4[i])
    Change.append(Momentum_4[i])
    Stock_Volatility.append(file4_close)
    Stock_Momentum.append(file4_Stock_Momentum)
    Index_Volatility.append(fileIndex_close)
    Index_Momentum.append(fileIndex_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_5[i])
    Change.append(Momentum_5[i])
    Stock_Volatility.append(file5_close)
    Stock_Momentum.append(file5_Stock_Momentum)
    Index_Volatility.append(fileIndex_close)
    Index_Momentum.append(fileIndex_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_Index[i])
    Change.append(Momentum_Index[i])
    Stock_Volatility.append(fileIndex_close)
    Stock_Momentum.append(fileIndex_Stock_Momentum)
    Index_Volatility.append(fileIndex_close)
    Index_Momentum.append(fileIndex_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

xl = pd.DataFrame({'Close': Close, 'Index_Momentum': Index_Momentum, 'Index_Volatility': Index_Volatility,
                   'Sector_Momentum': Sector_Momentum,
                   'Stock_Volatility': Stock_Volatility, 'Stock_Momentum': Stock_Momentum,
                   'Change': Change})

xl.to_csv("../dataset_modified_Input/Input_dataset.csv", index=False, header=True)


