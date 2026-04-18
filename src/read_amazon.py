import pandas as pd

df = pd.read_csv("C:/Users/Shivani Rao/Documents/fake-review-analyzer/data/archive/amazon_com-product_reviews__20200101_20200331_sample.csv")

print("Columns:\n", df.columns)

df = df[['Review Content']]
df = df.rename(columns={'Review Content': 'text'})

df = df.dropna()
df = df.drop_duplicates(subset='text')

print("\nSample:\n", df.head())
print("\nTotal reviews:", len(df))