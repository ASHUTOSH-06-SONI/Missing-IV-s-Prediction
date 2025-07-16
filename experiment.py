import pandas as pd
df = pd.read_csv("submission_wide_format.csv")
print(df.isnull().sum().sum())  # should print 0
