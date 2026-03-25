import builtins                           #overriding print statement globally to /n/n ending
def print(*args, **kwargs):
    kwargs.setdefault("end", "\n\n")
    return builtins.print(*args, **kwargs)


import pandas as pd   #understanding shape size head tail and basic info abt null-non null elements
df = pd.read_csv("creditcard.csv")
#print(df.shape) 
#print(df.head())
#print(df.tail())
#print(df.info())
#print(df.describe())
#print(df["Class"].value_counts(normalize = True)) #normalize converts raw no.s bw 0 and 1 in "%"
#print(df.duplicated().sum()) #total 101 duplicates found lets drop them!
df = df.drop_duplicates()


import matplotlib.pyplot as plt
#df['Amount'].hist(bins=50)
#plt.show() #Histogram showing amounts 

import numpy as np
df['Amount'] = np.log1p(df['Amount'])  #performing log transform

from sklearn.preprocessing import StandardScaler   
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
#df['Amount'].hist(bins=50)
#plt.show(), now its better

df["Hour"] = (df["Time"]/3600)%24
df["Hour"] = scaler.fit_transform(df[["Hour"]])
# df["Hour"].hist(bins=24)
# plt.show()
df.drop(columns=["Time"],inplace=True)

#corr = df.corr()

# import seaborn as sns
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12,10))
# sns.heatmap(corr, cmap='coolwarm')
# plt.show()

df.skew()