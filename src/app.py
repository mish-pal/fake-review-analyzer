import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import joblib
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK data if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# ---------- LOAD MODELS ---------- #
# We load the models once at startup
MODEL_DIR = "C:/Users/Shivani Rao/Documents/fake-review-analyzer/src/models"

@st.cache_resource
def load_models():
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "calibrated_model.pkl"))
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        explain_model = joblib.load(os.path.join(MODEL_DIR, "explain_model.pkl"))
        return model, vectorizer, explain_model
    except FileNotFoundError:
        return None, None, None

model, vectorizer, explain_model = load_models()


# ---------- CLEAN FUNCTION ---------- #
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


# ---------- FUNCTIONS ---------- #

def predict_review(review):
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    prob = model.predict_proba(vectorized)[0][1]
    return prob


def behavioral_score(review):
    """
    Returns
    -------
    score : float
        Behavioral adjustment score (negative for suspicious, positive for genuine)
    signals : dict
        A dictionary of the signals triggered
    """
    cleaned = clean_text(review)
    words = cleaned.split()
    if not words:
        return 0, {}
        
    unique_words = set(words)
    repetition_ratio = len(words) / (len(unique_words) + 1)
    
    strong_words = ["amazing", "best", "excellent", "perfect", "must", "love", "hate", "worst", "terrible", "awful"]
    emotion_count = sum(1 for w in words if w in strong_words)
    
    detail_score = len(words)
    
    # Multi-sentence structure
    sentences = re.split(r'[.!?]+', review)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Mixed sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(review)
    compound = sentiment['compound']
    mixed_sentiment = abs(compound) < 0.5 and (sentiment['pos'] > 0.1 and sentiment['neg'] > 0.1)
    
    # Domain-specific words
    domain_words = ["hotel", "room", "service", "food", "staff", "location", "price", "clean", "comfortable", "bed", "bathroom", "breakfast", "stay"]
    domain_count = sum(1 for w in words if w in domain_words)
    
    score = 0
    signals = {}
    
    # Penalties only if multiple signals
    penalty_count = 0
    if repetition_ratio > 1.5:
        penalty_count += 1
        signals['High Repetition'] = "High proportion of repeated words."
    if emotion_count > 2:
        penalty_count += 1
        signals['Excessive Emotion'] = "Overuse of strong emotive words."
    if detail_score < 10:
        penalty_count += 1
        signals['Low Detail'] = "Review lacks descriptive details."
        
    if penalty_count >= 2:
        score -= 20
        signals['Behavioral Penalty'] = "Multiple suspicious behavioral patterns detected (-20)."
    elif penalty_count == 1:
        # Don't penalize for a single signal, but track it
        signals['Notice'] = "Minor suspicious signal detected but not enough to penalize."
        
    # Positive signals
    if mixed_sentiment:
        score += 10
        signals['Mixed Sentiment'] = "Nuanced opinion indicating authenticity (+10)."
    if domain_count > 1:
        score += 5
        signals['Domain Relevance'] = "Uses natural context-specific vocabulary (+5)."
    if sentence_count > 2:
        score += 5
        signals['Detailed Structure'] = "Multi-sentence review provides better context (+5)."
        
    return score, signals


def get_authenticity(prob, review):
    # Base score: 100 means fully genuine, 0 means fully fake
    base = (1 - prob) * 100
    behavior_adj, signals = behavioral_score(review)
    
    # Ensure behavioral layer does not overpower ML output
    final_score = base + behavior_adj 
    final_score = max(0, min(100, final_score))
    
    # Confidence based on probability distance from 0.5
    confidence_dist = abs(prob - 0.5)
    if confidence_dist > 0.35:
        confidence = "High"
    elif confidence_dist > 0.15:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    # Asymmetric thresholding with safe zone
    if prob < 0.4:
        status = "Genuine"
    elif prob > 0.85:
        status = "Suspicious"
    else:
        status = "Uncertain"
        
    if status == "Genuine":
        if final_score >= 80:
            label = "Verified Authentic"
            color = "#28a745" # green
        else:
            label = "Likely Authentic"
            color = "#88b04b" # light green
    elif status == "Suspicious":
        if final_score <= 20:
            label = "Highly Suspicious"
            color = "#dc3545" # red
        else:
            label = "Potential Anomalies Detected"
            color = "#fd7e14" # orange
    else:
        label = "Requires Manual Verification"
        color = "#ffc107" # yellow

    return final_score, label, color, behavior_adj, signals, confidence, status


def get_top_words(review):
    cleaned = clean_text(review)
    vectorized = vectorizer.transform([cleaned])
    
    feature_names = vectorizer.get_feature_names_out()
    coefs = explain_model.coef_[0]
    
    words = []
    # Find nonzero entries in the sparse matrix
    for i in vectorized.nonzero()[1]:
        tfidf_val = vectorized[0, i]
        # Feature contribution = coefficient * tf-idf
        # Positive coefficient means it predicts "deceptive" (1)
        # Negative coefficient means it predicts "truthful" (0)
        contribution = coefs[i] * tfidf_val
        words.append((feature_names[i], contribution, tfidf_val))
    
    # Sort by absolute contribution to find the most influential words
    words = sorted(words, key=lambda x: abs(x[1]), reverse=True)
    return words[:5]

# ---------- UI ---------- #

st.set_page_config(page_title="Review Credibility System", layout="centered", initial_sidebar_state="collapsed")

# Styling
st.markdown("""
    <style>
    .big-score {
        font-size: 5rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        padding: 0;
        line-height: 1;
    }
    .status-badge {
        font-size: 1.3rem;
        font-weight: 600;
        text-align: center;
        padding: 10px;
        border-radius: 8px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        border-bottom: 2px solid #eee;
        padding-bottom: 5px;
        margin-top: 20px;
        margin-bottom: 10px;
        color: #444;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #f0f0f0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Review Credibility System")
st.markdown("Analyze review text for authenticity using a calibrated NLP model and behavioral heuristics.")

if model is None:
    st.error("Model files not found! Please run `python src/train_model.py` first.")
    st.stop()

review = st.text_area("Review Text:", height=150, placeholder="Paste a review here...")

if st.button("Evaluate Credibility", type="primary"):
    if not review.strip():
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing text..."):
            prob = predict_review(review)
            score, label, color, behavior_adj, signals, confidence, status = get_authenticity(prob, review)
            top_words = get_top_words(review)

        # Dashboard layout
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Row 1: Score & Status
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div class='section-header' style='margin-top: 0;'>Authenticity Score</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='big-score' style='color: {color};'>{int(score)}</div>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; margin-top: 10px; font-weight: bold;'>AI Confidence: <span style='color: #666;'>{confidence}</span></p>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='section-header' style='margin-top: 0;'>Analysis Result</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='status-badge' style='background-color: {color};'>{label}</div>", unsafe_allow_html=True)
            
            # Credibility Meter
            st.markdown("<p style='margin-bottom: 5px; font-weight: 500;'>Credibility Meter</p>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style="width:100%; background-color:#eee; border-radius:5px; padding:3px; margin-bottom: 5px;">
                    <div style="width:{score}%; background-color:{color}; padding:6px; border-radius:5px;"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:12px; color:gray;">
                    <span>Suspicious (0)</span><span>Uncertain (50)</span><span>Authentic (100)</span>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Row 2: Reasoning & Explainability
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("<div class='card' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header' style='margin-top: 0;'>📝 Behavioral Insights</div>", unsafe_allow_html=True)
            st.write(f"**ML Base Probability (Deceptive):** {prob*100:.1f}%")
            st.write(f"**Behavioral Adjustment:** {behavior_adj:+} pts")
            st.markdown("---")
            if signals:
                for sig, desc in signals.items():
                    if "High" in sig or "Excessive" in sig or "Low" in sig or "Penalty" in sig:
                        st.markdown(f"🔴 **{sig}**: {desc}")
                    elif "Notice" in sig:
                        st.markdown(f"🟡 **{sig}**: {desc}")
                    else:
                        st.markdown(f"🟢 **{sig}**: {desc}")
            else:
                st.write("No strong behavioral signals detected.")
            st.markdown("</div>", unsafe_allow_html=True)
                
        with col4:
            st.markdown("<div class='card' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header' style='margin-top: 0;'>🔍 Word Importance (ML)</div>", unsafe_allow_html=True)
            if top_words:
                st.markdown("<p style='font-size: 0.9em; color: #555;'>Words with the most influence on the probability score based on TF-IDF weight and model coefficient.</p>", unsafe_allow_html=True)
                for w, contrib, tfidf in top_words:
                    if contrib > 0:
                        direction = "Suspicious"
                        bar_color = "#dc3545"
                    else:
                        direction = "Authentic"
                        bar_color = "#28a745"
                    
                    # Scale width relative to max absolute contribution
                    max_c = max(abs(x[1]) for x in top_words) if top_words else 1
                    width = min((abs(contrib) / max_c) * 100, 100)
                    
                    st.markdown(f"""
                    <div style='margin-bottom: 12px;'>
                        <div style='display:flex; justify-content:space-between; font-size:14px; margin-bottom: 2px;'>
                            <strong>{w}</strong> <span style='color: {bar_color}; font-size: 0.9em;'>{direction}</span>
                        </div>
                        <div style="width:100%; background-color:#eee; height:8px; border-radius:4px;">
                            <div style="width:{width}%; background-color:{bar_color}; height:8px; border-radius:4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("Not enough text data to evaluate key words.")
            st.markdown("</div>", unsafe_allow_html=True)