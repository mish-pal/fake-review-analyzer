# 🛡️ ReviewGuard Pro - Fake Review Analyzer

ReviewGuard Pro is an advanced, data-driven web application designed to detect astroturfing, fake product reviews, and anomalous behavior in e-commerce datasets. It combines a highly polished modern user interface with a robust backend architecture, leveraging Natural Language Processing (NLP), Machine Learning, and Heuristic Analysis to determine the authenticity of textual data.

This project was built to address the growing issue of deceptive reviews by providing interpretable AI insights, bulk dataset processing, and live web interception.

---

## 🚀 Key Features

1. **🔍 Quantum Scan (Single Review Analysis)**: 
   - Deep forensic analysis of a single text input.
   - Provides an overall authenticity score, behavioral adjustments, and a confidence metric.
   - Features an Explainable AI (XAI) X-Ray view that highlights the specific lexical drivers (words) influencing the neural network's decision.

2. **📊 Multi-Batch Intel (Bulk CSV Ingestion)**:
   - Upload massive datasets (CSV files) to instantly map topological integrity.
   - Analyzes hundreds of reviews concurrently and returns aggregate metrics (e.g., mean authenticity score, anomalies detected).
   - Visualizes batch distribution using interactive Plotly pie charts and lists dominant behavioral flags across the dataset.

3. **🕸️ Live Intercept (Real-time Scraping)**:
   - Paste a live Amazon product URL.
   - The backend actively connects to the URL, bypasses basic protections, and scrapes active reviews.
   - Instantly routes the scraped data through the heuristic pipeline for immediate, real-world authenticity analysis.

4. **🌐 Community Intel (History)**:
   - A minimalist, persistent sidebar displaying a real-time history log of scanned reviews powered by an SQLite backend database.

---

## 🧠 Machine Learning Models & Architecture

The intelligence of ReviewGuard Pro relies on a hybrid pipeline combining machine learning and rule-based heuristics. 

### 1. NLP Text Processing & Vectorization
- **Text Cleaning**: Lowercasing, punctuation stripping, and standardizing text.
- **Advanced Tri-gram Architecture**: Uses `TfidfVectorizer` (Scikit-Learn) with up to 15,000 maximum features and n-grams (1, 3) to capture context and multi-word phrases, rather than just isolated words.

### 2. The Core Prediction Engine (Ensemble Model)
The primary predictive model (`calibrated_model.pkl`) is a highly robust **Optimal Hybrid Ensemble**, consisting of three distinct algorithms operating via a "soft" Voting Classifier:
- **Logistic Regression**: Excellent for high-dimensional sparse data like text.
- **Multinomial Naive Bayes**: Fast and reliable baseline for text classification.
- **Support Vector Classifier (SVC)**: Uses a linear kernel to establish strict decision boundaries between truthful and deceptive text.

To ensure the confidence scores displayed in the UI are statistically sound, this entire ensemble is wrapped in a **CalibratedClassifierCV** (using Platt scaling/sigmoid method).

### 3. Explainable AI (X-Ray Explainer)
To avoid the "black box" problem of AI, a parallel model (`explain_model.pkl`) is trained alongside the main ensemble:
- A dedicated **Logistic Regression** model is trained specifically to extract feature coefficients.
- During analysis, the system multiplies the TF-IDF vectors by these coefficients to determine exactly *which* words pushed the model toward predicting "Fake" or "Authentic". This creates the sub-lexical visual X-Ray in the UI.

### 4. Behavioral & Heuristic Scoring System
Raw ML probabilities are refined using a custom heuristic engine that evaluates human behavioral patterns:
- **Lexical Diversity / Repetition Ratio**: Flags bot-like repetition.
- **Sentiment Analysis**: Uses **NLTK's SentimentIntensityAnalyzer (VADER)** to check for nuanced mixed sentiment (a trait of real human reviews).
- **Gibberish Detection**: Cross-references vocabulary against the **NLTK English Words Corpus** to aggressively penalize automated spam or unrecognized character strings.
- **Emotional Density**: Flags reviews that excessively use extreme adjectives (e.g., "amazing", "worst", "terrible").
- **Domain Context**: Rewards reviews that use contextual vocabulary relevant to real-world experiences (e.g., "service", "price", "comfortable").

---

## 💻 Technology Stack

**Frontend:**
- **Streamlit**: Python web framework used for rapid UI development.
- **Custom CSS / Glassmorphism**: Heavy aesthetic modifications, radial gradients, modern fonts (Outfit, Space Grotesk), and micro-animations to create a premium feel.
- **Plotly Express**: Used for dynamic data visualization in batch and live processing.

**Backend & ML:**
- **FastAPI**: High-performance asynchronous backend framework that handles API routing (`/analyze`, `/analyze_batch`, `/scrape`).
- **Scikit-Learn**: Drives the entirety of the machine learning pipeline (Ensemble, TF-IDF, Calibration).
- **NLTK (Natural Language Toolkit)**: Used for behavioral analysis, sentiment tracking, and vocabulary validation.
- **BeautifulSoup4 & Requests**: Powers the live-scraping intercept engine.
- **SQLite**: Lightweight relational database used to store historical scans and community logs.
- **Pandas & NumPy**: For large-scale data manipulation and numerical operations.

---

## 🗣️ How to speak about this project (Review / Interview Notes)

When presenting this project, hit these key talking points:

1. **The Problem It Solves**: "I noticed that standard fake review detectors are 'black boxes'—they give you a score, but don't tell you *why*. My goal with ReviewGuard Pro was to build a system that is highly interpretable. It doesn't just catch astroturfing; it explains its reasoning."
2. **The Hybrid Approach**: "I didn't rely purely on a machine learning model. I built a hybrid pipeline. The core is an ensemble of Logistic Regression, Naive Bayes, and SVC, which catches statistical patterns. But I layered a custom Behavioral Heuristics engine on top of it. The heuristics check for things models often miss—like gibberish ratios, extreme emotional density, and nuanced mixed sentiment."
3. **Explainable AI (XAI)**: "One of the standout features is the 'X-Ray Scan'. I trained a parallel explainer model just to extract coefficients so the UI can visually show the user exactly which specific words drove the authenticity score."
4. **Real-World Applicability**: "It's not just a toy model for single text inputs. I built out a bulk CSV ingestion tool for data mapping, and more importantly, a Live Intercept feature using BeautifulSoup that can dynamically scrape Amazon links and run the data through the ML pipeline in real-time."
5. **Full-Stack Competency**: "I handled the entire lifecycle—from data cleaning and model training (Scikit-Learn), to building the asynchronous API (FastAPI), setting up the database (SQLite), and designing a highly polished, glassmorphism-inspired UI (Streamlit)."
