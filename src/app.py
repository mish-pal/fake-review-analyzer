import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import time

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="ReviewGuard Pro", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# Massive Custom CSS Overhaul: Glassmorphism and Aesthetic Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(18, 19, 28) 0%, rgb(4, 5, 8) 100%);
        color: #E2E8F0;
    }
    
    /* Header styling */
    h1 {
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        margin-bottom: 20px;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        padding: 24px;
        margin-bottom: 24px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 40px rgba(0, 198, 255, 0.2);
    }
    
    .big-score {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 6rem;
        font-weight: 700;
        text-align: center;
        line-height: 1;
        margin: 0;
        text-shadow: 0 0 20px rgba(255, 255, 255, 0.2);
    }
    
    .status-badge {
        font-size: 1.5rem;
        font-weight: 800;
        text-align: center;
        padding: 12px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Minimalist Sidebar */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 11, 20, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .review-card {
        padding: 16px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 16px;
        background: rgba(20, 22, 35, 0.6);
        color: #CBD5E1;
        transition: all 0.2s ease;
    }
    
    .review-card:hover {
        background: rgba(30, 32, 45, 0.8);
        border-color: rgba(0, 198, 255, 0.4);
    }

    /* X-Ray Progress Bars */
    .xray-bar-container {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
    }
    .xray-label {
        width: 120px;
        font-weight: 600;
        color: white;
        text-transform: capitalize;
    }
    .xray-track {
        flex-grow: 1;
        height: 10px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        position: relative;
        overflow: hidden;
    }
    .xray-fill {
        height: 100%;
        position: absolute;
        border-radius: 5px;
    }
    .xray-value {
        width: 80px;
        text-align: right;
        font-size: 0.8em;
        font-family: monospace;
        text-transform: uppercase;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# Define color logic for score rendering
def get_color(score, status):
    if "Authentic" in status:
        return "#00E676"  # Neon Green
    elif "Suspicious" in status or "Detected" in status:
        return "#FF1744"  # Neon Red
    else:
        return "#FFD600"  # Neon Yellow

# --- SIDEBAR: Community Insight & History ---
with st.sidebar:
    st.markdown("<h2 style='color: white; font-family: Space Grotesk;'>🌐 Community Intel</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Real-time history & verification network.</p>", unsafe_allow_html=True)
    
    if st.button("🔄 Sync Network"):
        pass
            
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
    
    try:
        res = requests.get(f"{API_BASE_URL}/reviews")
        if res.status_code == 200:
            reviews = res.json()
            if not reviews:
                st.write("No records found.")
            for r in reviews:
                total_votes = r['upvotes'] - r['downvotes']
                vote_color = "#00E676" if total_votes >= 0 else "#FF1744"
                
                status_guess = "Authentic" if r['final_score'] >= 50 else "Suspicious"
                
                st.markdown(f"""
                <div class='review-card'>
                    <div style='font-size: 0.75em; color: #64748B; margin-bottom: 8px;'>SECURE LOG: {r['created_at'][:16]}</div>
                    <div style='font-size: 0.95em; margin-bottom: 12px; font-style: italic;'>" {r['text'][:65]}..."</div>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <strong style='color:{get_color(r["final_score"], status_guess)}; font-size: 1.1em;'>[{int(r['final_score'])}/100]</strong>
                        <span style='color:{vote_color}; font-weight:800; font-family: monospace;'>VOTES: {total_votes}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("⬆️ Verify", key=f"up_{r['id']}", use_container_width=True):
                        requests.post(f"{API_BASE_URL}/vote", json={"review_id": r['id'], "vote": "up"})
                        st.rerun()
                with cc2:
                    if st.button("⬇️ Contest", key=f"down_{r['id']}", use_container_width=True):
                        requests.post(f"{API_BASE_URL}/vote", json={"review_id": r['id'], "vote": "down"})
                        st.rerun()
                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        else:
            st.error("API Error.")
    except requests.exceptions.ConnectionError:
        st.error("Backend Server is unreachable.")

# --- MAIN CONTENT ---
st.title("ReviewGuard Pro")
st.markdown("<p style='font-size: 1.2rem; color: #94A3B8; margin-top: -15px; margin-bottom: 30px;'>Advanced Forensics & NLP Astroturfing Detection</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Quantum Scan", "📊 Multi-Batch Intel", "🌐 Live Intercept (Beta)"])

with tab1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    review = st.text_area("Input forensic text string for deep analysis:", height=150, placeholder="Paste a suspect review here...")

    if st.button("Execute X-Ray Scan", type="primary", use_container_width=True):
        if not review.strip():
            st.warning("Data payload cannot be empty.")
        elif len(review.split()) < 5:
            st.warning("Insufficient structural data. Provide at least 5 words.")
        else:
            with st.spinner("Executing neural heuristics..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/analyze", json={"review": review})
                    if response.status_code == 200:
                        data = response.json()
                        score = data["score"]
                        status = data["status"]
                        color = get_color(score, status)
                        confidence = data["confidence"]
                        behavior_adj = data["behavior_adjustment"]
                        signals = data["reasoning"]
                        top_words = data.get("top_words", [])

                        # Row 1: Score & Status
                        col1, col2 = st.columns([1, 1.5])
                        
                        with col1:
                            st.markdown(f"""
                                <div class='glass-card' style='text-align: center;'>
                                    <p style='color: #94A3B8; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;'>Authenticity Score</p>
                                    <div class='big-score' style='color: {color}; text-shadow: 0 0 30px {color}88;'>{int(score)}</div>
                                    <p style='margin-top: 15px; font-weight: 600; color: white;'>Confidence: <span style='color: #00c6ff;'>{confidence}</span></p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown(f"""
                                <div class='glass-card'>
                                    <p style='color: #94A3B8; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;'>Threat Level</p>
                                    <div class='status-badge' style='background-color: {color}; box-shadow: 0 0 20px {color}66;'>{status}</div>
                                    <p style='color: #CBD5E1; margin-bottom: 5px;'>Behavioral Alignment: <strong style='color: {"#00E676" if behavior_adj >= 0 else "#FF1744"}'>{behavior_adj:+} pts</strong></p>
                            """, unsafe_allow_html=True)
                            
                            for sig in signals:
                                icon = "🔴" if "penalty" in sig.lower() or "suspicion" in sig.lower() or "low detail" in sig.lower() or "bias" in sig.lower() else "🟢"
                                st.markdown(f"<p style='font-size: 0.9em; margin: 2px 0;'>{icon} {sig}</p>", unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)

                        # Row 2: Explainable AI (X-Ray)
                        if top_words:
                            st.markdown("""
                            <div class='glass-card'>
                                <h3 style='font-family: Space Grotesk; margin-bottom: 20px;'>🧬 Model X-Ray: Sub-Lexical Analysis</h3>
                                <p style='color: #94A3B8; font-size: 0.9em; margin-bottom: 20px;'>Visualizing the isolated mathematical vectors driving the neural net decision.</p>
                            """, unsafe_allow_html=True)
                            
                            for tw in top_words:
                                word = tw['word']
                                cont = tw['contribution']
                                
                                width = min(abs(cont) * 20, 100) # Arbitrary scaling for UI beauty
                                bar_color = "#FF1744" if cont > 0 else "#00E676" # Because model predicts DECEPTIVE (1), positive cont == fake
                                
                                label = "Suspicious (Fake)" if cont > 0 else "Authentic"
                                
                                st.markdown(f"""
                                <div class='xray-bar-container'>
                                    <div class='xray-label'>{word}</div>
                                    <div class='xray-track'>
                                        <div class='xray-fill' style='width: {width}%; background-color: {bar_color}; box-shadow: 0 0 10px {bar_color};'></div>
                                    </div>
                                    <div class='xray-value' style='color: {bar_color}'>{label}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"Integrity check failed: {response.json()}")
                except Exception as e:
                    st.error(f"Comms Relays Offline: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🗂️ Macro-Scale CSV Ingestion")
    st.markdown("<p style='color: #94A3B8;'>Map large dataset topologies instantly.</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV Architecture", type=["csv"], key="csv_upload")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        st.markdown("<div style='background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px;'>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        col_select = st.selectbox("Designate target vector (Text Column):", df.columns)
        limit = st.slider("Max Execution Subroutines", 10, 1000, 100)
        
        if st.button("Execute Batch Scan", type="primary"):
            reviews_list = df[col_select].dropna().astype(str).tolist()[:limit]
            
            with st.spinner(f"Processing vector mappings for {len(reviews_list)} items..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/analyze_batch", json={"reviews": reviews_list})
                    if response.status_code == 200:
                        data = response.json()
                        metrics = data["metrics"]
                        results = data["results"]
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Items Processed", metrics['total_analyzed'])
                        m2.metric("Mean Authenticity Score", f"{metrics['average_score']}/100")
                        m3.metric("Anomalies Detected", metrics['suspicious_count'])
                        
                        st.markdown("---")
                        
                        df_res = pd.DataFrame([{"Status": "Authentic", "Count": metrics['authentic_count']},
                                              {"Status": "Suspicious (Anomaly)", "Count": metrics['suspicious_count']}])
                        
                        fig = px.pie(df_res, values='Count', names='Status', title="Topological Integrity",
                                     color="Status", color_discrete_map={"Authentic": "#00E676", "Suspicious (Anomaly)": "#FF1744"}, hole=0.7)
                        
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("API error during batch sequence.")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("### 🕸️ Live Subnet Intercept")
    st.markdown("<p style='color: #94A3B8;'>Simulated endpoint for Amazon URL live scraping and instant heuristic pipeline routing.</p>", unsafe_allow_html=True)
    
    url_input = st.text_input("Target URL (Amazon Product Link):", placeholder="https://www.amazon.com/dp/B08XJG8KV...")
    
    if st.button("Intercept & Analyze", type="primary"):
        if not any(domain in url_input.lower() for domain in ["amazon", "amzn"]):
            st.error("Target verification failed. Please enter a valid Amazon product URL.")
        else:
            with st.spinner("Connecting to root domain..."):
                time.sleep(1.5)
            with st.spinner("Bypassing captchas & parsing HTML tree..."):
                time.sleep(1.5)
            with st.spinner("Fetching live review strings..."):
                time.sleep(1)
                
            st.success("Target successfully scraped! Found 100 recent live reviews.")
            
            st.markdown("<div class='glass-card' style='margin-top: 20px;'>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Live Items Captured", 100)
            m2.metric("Mean URL Authenticity", "34.5/100")
            m3.metric("Critical Astroturfing Flags", 65)
            
            df_mock = pd.DataFrame([{"Status": "Authentic", "Count": 35},
                                  {"Status": "Deceptive Farm", "Count": 65}])
            
            fig2 = px.pie(df_mock, values='Count', names='Status', title="Live Data Authenticity",
                         color="Status", color_discrete_map={"Authentic": "#00E676", "Deceptive Farm": "#FF1744"}, hole=0.6)
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='white')
            
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("<p style='color: #FF1744; text-align: center; font-weight: bold;'>⚠️ WARNING: Product exhibits highly suspicious review patterns indicative of widespread bot astroturfing.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
    st.markdown("</div>", unsafe_allow_html=True)