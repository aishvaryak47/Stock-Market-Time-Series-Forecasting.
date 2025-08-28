import pandas as pd

df = pd.read_csv("data/RELIANCE.NS.csv")
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head(10))
print(df.tail(10))
