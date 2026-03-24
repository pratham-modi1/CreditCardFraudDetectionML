import builtins                           #overriding print statement globally to /n/n ending
def print(*args, **kwargs):
    kwargs.setdefault("end", "\n\n")
    return builtins.print(*args, **kwargs)


import pandas as pd
df = pd.read_csv("creditcard.csv")
print(df.shape) 
print(df.head())
print(df.tail())
print(df.info())
print(df["Class"].value_counts())