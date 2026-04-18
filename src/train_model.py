import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Paths
BASE_DIR = "C:/Users/Shivani Rao/Documents/fake-review-analyzer"
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

# We want real > fake approx 2:1 ratio
num_fake = len(df_dec[df_dec['label'] == 1])
desired_real_total = num_fake * 2 
existing_real = len(df_dec[df_dec['label'] == 0])
needed_amazon_real = desired_real_total - existing_real

if needed_amazon_real > 0:
    df_amz_sampled = df_amz.sample(n=min(needed_amazon_real, len(df_amz)), random_state=42)
else:
    df_amz_sampled = pd.DataFrame(columns=['text', 'label'])

# Merge and shuffle
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

# 4. Vectorize
print("Applying TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# 5. Train & Evaluate Models
def evaluate(model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    cm = confusion_matrix(y, preds)
    # TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return acc, prec, rec, f1, fpr, cm

print("\n--- Logistic Regression (Baseline) ---")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
lr_acc, lr_prec, lr_rec, lr_f1, lr_fpr, lr_cm = evaluate(lr, X_test, y_test)
print(f"Accuracy:  {lr_acc:.4f}")
print(f"Precision: {lr_prec:.4f}")
print(f"Recall:    {lr_rec:.4f}")
print(f"F1-score:  {lr_f1:.4f}")
print(f"FPR:       {lr_fpr:.4f}")
print(f"Conf Mat:\n{lr_cm}")

print("\n--- Multinomial NB ---")
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_acc, nb_prec, nb_rec, nb_f1, nb_fpr, nb_cm = evaluate(nb, X_test, y_test)
print(f"Accuracy:  {nb_acc:.4f}")
print(f"Precision: {nb_prec:.4f}")
print(f"Recall:    {nb_rec:.4f}")
print(f"F1-score:  {nb_f1:.4f}")
print(f"FPR:       {nb_fpr:.4f}")
print(f"Conf Mat:\n{nb_cm}")


print("\n--- Calibrated Logistic Regression ---")
base_lr = LogisticRegression(max_iter=1000, random_state=42)
calibrated_model = CalibratedClassifierCV(base_lr, method='sigmoid', cv=5)
calibrated_model.fit(X_train, y_train)
cal_acc, cal_prec, cal_rec, cal_f1, cal_fpr, cal_cm = evaluate(calibrated_model, X_test, y_test)
print(f"Accuracy:  {cal_acc:.4f}")
print(f"Precision: {cal_prec:.4f}")
print(f"Recall:    {cal_rec:.4f}")
print(f"F1-score:  {cal_f1:.4f}")
print(f"FPR:       {cal_fpr:.4f}")
print(f"Conf Mat:\n{cal_cm}")

# Ensure we save models
print("\nSaving models...")
joblib.dump(calibrated_model, os.path.join(MODEL_DIR, "calibrated_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

# Save the base LR model for explainability (since calibrated model doesn't easily expose coef_)
joblib.dump(lr, os.path.join(MODEL_DIR, "explain_model.pkl"))
print(f"Models saved in {MODEL_DIR}")