import pandas as pd

# Tao ra 1 series voi mot list bat ki

s = pd.Series(['Nam', 37, -110122017, 'KHMT'])
print(type(s))
print(s)

s = pd.Series(['Nam', 37, -110122017, 'KHMT'], index=['B', 'O', 'N', 'G'])
print(type(s))
print(s)