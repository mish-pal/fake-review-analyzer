import pandas as pd
import numpy as np
import os
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DECEPTIVE_PATH = os.path.join(BASE_DIR, "data/archive/deceptive-opinion.csv")
AMAZON_PATH = os.path.join(BASE_DIR, "data/archive/amazon_com-product_reviews__20200101_20200331_sample.csv")
MODEL_DIR = os.path.join(BASE_DIR, "src/models")

os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Load Data
print("Loading datasets...")
df_dec = pd.read_csv(DECEPTIVE_PATH)
df_dec['label'] = df_dec['deceptive'].map({'truthful': 0, 'deceptive': 1})
df_dec = df_dec[['text', 'label']]

df_amz = pd.read_csv(AMAZON_PATH)
df_amz = df_amz.rename(columns={'Review Content': 'text'}).dropna(subset=['text'])
df_amz['label'] = 0
df_amz = df_amz[['text', 'label']]

num_fake = len(df_dec[df_dec['label'] == 1])
desired_real_total = num_fake * 2 
existing_real = len(df_dec[df_dec['label'] == 0])
needed_amazon_real = desired_real_total - existing_real

if needed_amazon_real > 0:
    df_amz_sampled = df_amz.sample(n=min(needed_amazon_real, len(df_amz)), random_state=42)
else:
    df_amz_sampled = pd.DataFrame(columns=['text', 'label'])

df = pd.concat([df_dec, df_amz_sampled], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Dataset Distribution:")
print(df['label'].value_counts())

# 2. Text Cleaning
print("Cleaning text...")
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

# 3. Split BEFORE vectorization
print("Splitting datasets...")
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# 4. Vectorize Text
print("Vectorizing Text with Advanced Tri-gram Architecture...")
vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# 5. Train Ensemble Model
print("\n--- Training Optimal Hybrid Ensemble ---")
lr = LogisticRegression(C=2.0, max_iter=2000, random_state=42)
nb = MultinomialNB(alpha=0.5)
svc = SVC(kernel='linear', probability=True, random_state=42, C=1.0)

voting_clf = VotingClassifier(
    estimators=[('lr', lr), ('nb', nb), ('svc', svc)],
    voting='soft'
)

calibrated_model = CalibratedClassifierCV(voting_clf, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)

# Train a dedicated explainer model on the full dataset for UI interpretation
print("Training X-Ray Explainer...")
explain_model = LogisticRegression(max_iter=1000, random_state=42)
explain_model.fit(X_train, y_train)

def evaluate(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return acc, prec, rec, f1, fpr, cm

cal_acc, cal_prec, cal_rec, cal_f1, cal_fpr, cal_cm = evaluate(calibrated_model, X_test, y_test)

print(f"Ensemble Accuracy:  {cal_acc:.4f}")
print(f"Ensemble Precision: {cal_prec:.4f}")
print(f"Ensemble Recall:    {cal_rec:.4f}")
print(f"Ensemble F1-score:  {cal_f1:.4f}")
print(f"Ensemble FPR:       {cal_fpr:.4f}")

# Ensure we save models
print("\nSaving robust models...")
joblib.dump(calibrated_model, os.path.join(MODEL_DIR, "calibrated_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(explain_model, os.path.join(MODEL_DIR, "explain_model.pkl"))

# Cleanup of old broken transformer models
try:
    os.remove(os.path.join(MODEL_DIR, "transformer_classifier.pkl"))
except FileNotFoundError:
    pass

print(f"Core models saved perfectly in {MODEL_DIR}")