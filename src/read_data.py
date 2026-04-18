import pandas as pd

# load dataset (correct path)
df = pd.read_csv("C:/Users/Shivani Rao/Documents/ML-project/data/archive/deceptive-opinion.csv")

# show first 5 rows
print(df.head())

# show column names
print("\nColumns:", df.columns)