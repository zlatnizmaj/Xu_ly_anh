import pandas as pd
path_files_CSV = "./dataset_original"
file = "../dataset_original/AAPL.CSV"

xl = pd.read_csv(file)
date = xl['Date']
Close = xl['Close']
AD_Position = Close.copy()
Change = [0] * len(date)
Momentum = [0] * len(date)

dates = [i.split(' ', 1)[0] for i in date]

reference = dates[0]


for i in range(1, len(date)):
    if Close[i] > Close[i-1] :
        Momentum[i] = "1"
        Change[i] = (Close[i]-Close[i-1])/Close[i-1]
    else :
        Momentum[i] = "0"
        Change[i] = (Close[i-1] - Close[i]) / Close[i - 1]

xl = pd.DataFrame({'Change': Change, 'Close': Close, 'Date': date, 'Momentum': Momentum})
xl.to_csv("../dataset_modified_Input/Apple_modified.csv", index=False, header=True)
