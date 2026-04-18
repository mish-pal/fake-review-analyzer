import pandas as pd
import re

# load dataset
df = pd.read_csv("C:/Users/Shivani Rao/Documents/ML-project/data/archive/deceptive-opinion.csv")

# function to clean text
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation
    return text

# apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# show result
print(df[['text', 'clean_text']].head())