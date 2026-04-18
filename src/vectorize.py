import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# load dataset
df = pd.read_csv("C:/Users/Shivani Rao/Documents/ML-project/data/archive/deceptive-opinion.csv")

# clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['clean_text'])

# labels (convert to 0 and 1)
y = df['deceptive'].map({'truthful': 0, 'deceptive': 1})

print("Shape of X:", X.shape)
print("Sample features:", vectorizer.get_feature_names_out()[:10])