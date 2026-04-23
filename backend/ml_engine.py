import numpy as np
import re
import os
import joblib
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Dict, Any, List
import nltk

try:
    from nltk.corpus import words as nltk_words
    ENGLISH_VOCAB = set(nltk_words.words())
except LookupError:
    nltk.download('words', quiet=True)
    from nltk.corpus import words as nltk_words
    ENGLISH_VOCAB = set(nltk_words.words())


MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'models'))

try:
    model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    explain_model = joblib.load(os.path.join(MODEL_DIR, "explain_model.pkl"))
except Exception as e:
    print(f"Warning: Models not found in {MODEL_DIR}. Please run training script. Error: {e}")
    model = None
    vectorizer = None
    explain_model = None

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def behavioral_score(review: str):
    cleaned = clean_text(review)
    words = cleaned.split()
    if not words:
        return 0, []
        
    unique_words = set(words)
    repetition_ratio = len(words) / (len(unique_words) + 1)
    
    strong_words = ["amazing", "best", "excellent", "perfect", "must", "love", "hate", "worst", "terrible", "awful"]
    emotion_count = sum(1 for w in words if w in strong_words)
    
    detail_score = len(words)
    sentences = re.split(r'[.!?]+', review)
    sentence_count = len([s for s in sentences if s.strip()])
    
    try:
        sia = SentimentIntensityAnalyzer()
        sentiment = sia.polarity_scores(review)
        compound = sentiment['compound']
        mixed_sentiment = abs(compound) < 0.5 and (sentiment['pos'] > 0.1 and sentiment['neg'] > 0.1)
    except Exception:
        mixed_sentiment = False
    
    domain_words = ["hotel", "room", "service", "food", "staff", "location", "price", "clean", "comfortable", "bed", "bathroom", "breakfast", "stay"]
    domain_count = sum(1 for w in words if w in domain_words)
    
    score = 0
    reasoning = []
    
    penalty_count = 0
    if repetition_ratio > 1.5:
        penalty_count += 1
        reasoning.append("High repetition detected, which is unusual for authentic reviews.")
    if emotion_count > 2:
        penalty_count += 1
        reasoning.append("Excessive use of strong emotive words, potentially indicating bias or astroturfing.")
    if detail_score < 10:
        penalty_count += 1
        reasoning.append("Low detail profile; the review lacks descriptive substance.")
        
    gibberish_count = 0
    for w in words:
        if len(w) > 2 and w not in ENGLISH_VOCAB and not w.isdigit():
            # also allow domain words
            if w not in domain_words and w not in strong_words:
                gibberish_count += 1
                
    gibberish_ratio = gibberish_count / len(words) if words else 0
    if gibberish_ratio > 0.3:
        score -= 50
        reasoning.append(f"High anomaly detection ({int(gibberish_ratio*100)}% unrecognized vocabulary). Likely gibberish or automated spam.")
        
    if penalty_count >= 2:
        score -= 20
        reasoning.insert(0, "Multiple suspicious behavioral patterns triggered a severe penalty.")
    
    if mixed_sentiment:
        score += 10
        reasoning.append("Review contains mixed sentiment, characteristic of nuanced authentic opinions.")
    if domain_count > 1:
        score += 5
        reasoning.append("Appropriate use of contextual domain-specific vocabulary.")
    if sentence_count > 2:
        score += 5
        reasoning.append("Multi-sentence structure provides solid foundational context.")
        
    return score, reasoning

def analyze_review(review: str) -> Dict[str, Any]:
    if not model or not vectorizer or not explain_model:
        raise ValueError("ML models were not loaded properly.")
        
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    prob = float(model.predict_proba(vectorized)[0][1])
    
    base = (1 - prob) * 100
    behavior_adj, reasoning = behavioral_score(review)
    
    if any("unrecognized vocabulary" in r for r in reasoning):
        base = min(base, 30)
        prob = max(prob, 0.9)
        
    final_score = float(max(0, min(100, base + behavior_adj)))
    
    confidence_dist = abs(prob - 0.5)
    if confidence_dist > 0.35:
        confidence = "High"
    elif confidence_dist > 0.15:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    if prob < 0.4:
        status_base = "Genuine"
    elif prob > 0.85:
        status_base = "Suspicious"
    else:
        status_base = "Uncertain"
        
    if status_base == "Genuine":
        if final_score >= 80:
            status = "Verified Authentic"
        else:
            status = "Likely Authentic"
    elif status_base == "Suspicious":
        if final_score <= 20:
            status = "Highly Suspicious"
        else:
            status = "Potential Anomalies Detected"
    else:
        status = "Requires Manual Verification"

    # XAI Extraction
    feature_names = vectorizer.get_feature_names_out()
    coefs = explain_model.coef_[0]
    
    words = []
    for i in vectorized.nonzero()[1]:
        tfidf_val = vectorized[0, i]
        contribution = coefs[i] * tfidf_val
        words.append({
            "word": str(feature_names[i]),
            "contribution": float(round(contribution, 4)),
            "tfidf": float(round(tfidf_val, 4))
        })
    
    top_words = sorted(words, key=lambda x: abs(x["contribution"]), reverse=True)[:5]

    return {
        "score": round(final_score, 2),
        "status": status,
        "confidence": confidence,
        "behavior_adjustment": behavior_adj,
        "top_words": top_words,
        "reasoning": reasoning
    }

def analyze_batch(reviews: List[str]) -> Dict[str, Any]:
    if not reviews:
        return {"results": [], "metrics": {}}
        
    if not model or not vectorizer or not explain_model:
        raise ValueError("ML models not loaded.")

    cleaned_texts = [clean_text(r) for r in reviews]
    vectors = vectorizer.transform(cleaned_texts)
    probs = model.predict_proba(vectors)[:, 1]
    
    results = []
    auth_count = 0
    susp_count = 0
    total_score = 0
    
    all_reasons = {}
    
    for i, review in enumerate(reviews):
        # We can run basic behavior score over each
        behavior_adj, reasoning = behavioral_score(review)
        base = (1 - probs[i]) * 100
        
        if any("unrecognized vocabulary" in r for r in reasoning):
            base = min(base, 30)
            
        final_score = float(max(0, min(100, base + behavior_adj)))
        
        status = "Authentic" if final_score >= 50 else "Suspicious"
        if status == "Authentic": auth_count += 1
        else: susp_count += 1
        
        total_score += final_score
        
        for r in reasoning:
            all_reasons[r] = all_reasons.get(r, 0) + 1
            
        results.append({
            "text": review,
            "score": round(final_score, 2),
            "status": status
        })
        
    # Get top common behavioral reasons
    common_reasons = sorted(all_reasons.items(), key=lambda x: x[1], reverse=True)[:4]
    formatted_reasons = [f"{k} ({v} occurrences)" for k, v in common_reasons]
    
    # Extract overall XAI for batch
    # Sum the tfidf vectors across the batch
    sum_vectors = vectors.sum(axis=0)
    feature_names = vectorizer.get_feature_names_out()
    coefs = explain_model.coef_[0]
    
    words = []
    # nonzero elements in the summed vector
    import numpy as np
    sum_vectors_np = np.asarray(sum_vectors)
    for i in np.nonzero(sum_vectors_np)[1]:
        tfidf_val = float(sum_vectors_np[0, i])
        contribution = coefs[i] * tfidf_val
        words.append({
            "word": str(feature_names[i]),
            "contribution": float(round(contribution, 4)),
            "tfidf_sum": float(round(tfidf_val, 4))
        })
    
    top_batch_words = sorted(words, key=lambda x: abs(x["contribution"]), reverse=True)[:8]

    metrics = {
        "total_analyzed": len(reviews),
        "authentic_count": auth_count,
        "suspicious_count": susp_count,
        "average_score": round(total_score / len(reviews), 2)
    }
    
    return {
        "results": results, 
        "metrics": metrics,
        "common_reasons": formatted_reasons,
        "top_batch_words": top_batch_words
    }
