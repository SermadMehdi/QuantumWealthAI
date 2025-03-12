import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import requests
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import neal
from transformers import pipeline
import streamlit as st
import plotly.express as px
from functools import lru_cache
import concurrent.futures

# ------------------------------------------------------------
#  API CONFIGURATION (REMOVED FOR SECURITY - ADD YOUR OWN)
# ------------------------------------------------------------
API_URL = "YOUR_API_URL_HERE"
API_CODE = "YOUR_API_KEY_HERE"

# ------------------------------------------------------------
#  OPTIMIZED FUNCTION: LOAD HISTORICAL STOCK DATA (CACHED)
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data(stock_name, start_date, end_date, retries=1):
    params = {
        "code": API_CODE,
        "name": stock_name,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d")
    }
    for attempt in range(retries + 1):
        try:
            response = requests.get(API_URL, params=params)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return pd.DataFrame(data)
                else:
                    st.error(f"No data returned for {stock_name}.")
                    return None
            else:
                st.error(f"Error fetching data for {stock_name}: {response.status_code} {response.text}")
                if attempt < retries:
                    st.info(f"Retrying {stock_name} (attempt {attempt+1})...")
        except Exception as e:
            st.error(f"Exception fetching data for {stock_name}: {e}")
    return None

# ------------------------------------------------------------
#  FUNCTION: CALCULATE EXPECTED RETURN AND VOLATILITY
# ------------------------------------------------------------
def calculate_metrics(df):
    if df is None or df.empty or 'Close' not in df.columns:
        return None, None
    
    if 'date' in df.columns:
        df = df.sort_values(by='date')
    elif 'Date' in df.columns:
        df = df.sort_values(by='Date')

    df['return'] = df['Close'].pct_change()
    expected_return = (df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1
    volatility = df['return'].std()
    return expected_return, volatility

# ------------------------------------------------------------
#  OPTIMIZED FUNCTION: ANALYZE MARKET SENTIMENT (BATCH PROCESSING)
# ------------------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english", 
        revision="714eb0f",
        framework="pt"
    )

def analyze_market_sentiment_from_csv(filename="news.csv", num_rows=200):
    try:
        encodings = ["utf-8", "latin-1", "ISO-8859-1"]
        for encoding in encodings:
            try:
                news_df = pd.read_csv(filename, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        
        news_df = news_df.head(num_rows)
        if news_df.shape[1] < 2:
            st.error("CSV must have at least two columns (sentiment label, news text).")
            return 0.0

        sentiment_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
        sentiment_column = news_df.columns[0]
        text_column = news_df.columns[1]
        news_df[sentiment_column] = news_df[sentiment_column].str.lower().map(sentiment_map)

        sentiment_pipeline = load_sentiment_model()
        texts = news_df[text_column].dropna().tolist()
        
        results = sentiment_pipeline(texts, batch_size=16)
        
        hf_scores = []
        for result in results:
            label = result['label']
            score = result['score']
            if label.upper() == 'POSITIVE':
                adjusted_score = 0.5 + score
            elif label.upper() == 'NEGATIVE':
                adjusted_score = -0.5 - score
            else:
                adjusted_score = 0.0
            hf_scores.append(adjusted_score)

        avg_label_sentiment = np.mean(news_df[sentiment_column].dropna().values) if not news_df[sentiment_column].dropna().empty else 0.0
        avg_hf_sentiment = np.mean(hf_scores) if hf_scores else 0.0

        hybrid_sentiment_score = (0.7 * avg_label_sentiment) + (0.3 * avg_hf_sentiment)
        return hybrid_sentiment_score if not np.isnan(hybrid_sentiment_score) else 0.0

    except Exception as e:
        st.error(f"Error processing news.csv: {e}")
        return 0.0

# ------------------------------------------------------------
#  FUNCTION: BUILD QUBO MODEL FOR PORTFOLIO OPTIMIZATION
# ------------------------------------------------------------
def build_qubo_model(assets, k=5, sentiment_adjustment=0.0):
    Q = {}
    n = len(assets)
    risk_factor = 1.0
    return_factor = 1.0
    returns = [asset['expected_return'] + sentiment_adjustment for asset in assets]
    risks = [asset['volatility'] for asset in assets]
    A = max(returns) * 10  
    for i in range(n):
        Q[(i, i)] = risk_factor * risks[i] - return_factor * returns[i] + A * (1 - 2 * k)
        for j in range(i+1, n):
            Q[(i, j)] = 2 * A
    return Q

# ------------------------------------------------------------
#  FUNCTION: SOLVE THE QUBO PROBLEM
# ------------------------------------------------------------
def solve_qubo(Q, num_reads=100):
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=num_reads)
    return sampleset.first.sample

# ------------------------------------------------------------
#  FUNCTION: ALLOCATE FUNDS BASED ON RISK-RETURN RATIO
# ------------------------------------------------------------
def allocate_funds(selected_assets, total_investment):
    epsilon = 1e-6
    scores = np.array([
        asset['expected_return'] / (asset['volatility'] + epsilon)
        for asset in selected_assets
    ])
    min_score = np.min(scores)
    if min_score < 0:
        scores = scores - min_score
    if np.sum(scores) == 0:
        weights = np.ones(len(selected_assets)) / len(selected_assets)
    else:
        weights = scores / np.sum(scores)
    allocated_amounts = total_investment * weights
    return weights, allocated_amounts

# ------------------------------------------------------------
#  MAIN APPLICATION (PREMIUM UI/UX)
# ------------------------------------------------------------
def main():
    # Premium CSS Styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #ffffff;
            padding: 2rem 2rem 4rem;
        }
        
        .stAlert {
            border-left: 4px solid #6366f1;
            background: rgba(99, 102, 241, 0.1) !important;
            border-radius: 12px;
        }
        
        .metric-card {
            padding: 1.5rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 48px rgba(0, 0, 0, 0.2);
        }
        
        .stSlider > div {
            background: rgba(255, 255, 255, 0.05) !important;
            border-radius: 12px;
            padding: 10px;
        }
        
        .stNumberInput input {
            background: rgba(255, 255, 255, 0.05) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
        }
        
        .header-gradient {
            background: linear-gradient(45deg, #6366f1, #a855f7, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: gradient-animation 6s ease infinite;
        }
        
        @keyframes gradient-animation {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .neon-border {
            position: relative;
            border-radius: 20px;
            overflow: hidden;
        }
        
        .neon-border::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: 20px;
            padding: 2px;
            background: linear-gradient(45deg, #6366f1, #a855f7);
            -webkit-mask: linear-gradient(#fff 0 0) content-box, 
                          linear-gradient(#fff 0 0);
            mask: linear-gradient(#fff 0 0) content-box, 
                  linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            animation: gradient-animation 6s ease infinite;
        }
    </style>
    """, unsafe_allow_html=True)

    # Premium Header Section
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0 4rem;">
        <h1 class="header-gradient" style="font-size: 3rem; margin-bottom: 1rem; letter-spacing: -0.03em;">
            🌌 QuantumWealth AI
        </h1>
        <h3 style="color: #94a3b8; font-weight: 400; letter-spacing: 0.05em;">
            Next-Generation Portfolio Optimization Engine
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # Premium Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="neon-border" style="padding: 1.5rem; border-radius: 20px;">
            <h2 style="color: #818cf8; margin-bottom: 2rem; font-weight: 600;">⚡ Control Panel</h2>
        """, unsafe_allow_html=True)
        
        investment_amount = st.number_input(
            "Investment Capital ($)",
            min_value=1000, 
            value=20000,
            step=1000,
            format="%d"
        )
        
        k = st.slider(
            "Asset Diversification Level", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Optimal risk distribution configuration"
        )
        
        st.markdown("""
            <div style="margin-top: 2rem; color: #94a3b8;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin: 1rem 0;">
                    <div style="width: 8px; height: 8px; background: #6366f1; border-radius: 50%;"></div>
                    <div>Real-time Market Data</div>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem; margin: 1rem 0;">
                    <div style="width: 8px; height: 8px; background: #a855f7; border-radius: 50%;"></div>
                    <div>Quantum Annealing Engine</div>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem; margin: 1rem 0;">
                    <div style="width: 8px; height: 8px; background: #ec4899; border-radius: 50%;"></div>
                    <div>Neural Sentiment Analysis</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Data Loading with Progress
    with st.spinner("🌌 Initializing Quantum Financial Matrix..."):
        stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NFLX", "NVDA"]
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()

        assets = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(load_stock_data, stock, start_date, end_date): stock 
                for stock in stocks
            }
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            processed_count = 0
            
            for future in concurrent.futures.as_completed(futures):
                stock = futures[future]
                processed_count += 1
                try:
                    df = future.result()
                    if df is not None:
                        expected_return, volatility = calculate_metrics(df)
                        if expected_return is not None and volatility is not None:
                            assets.append({
                                "symbol": stock,
                                "expected_return": expected_return,
                                "volatility": volatility
                            })
                    status_text.markdown(f"<div style='color: #94a3b8;'>📡 Receiving *{stock}* data ({processed_count}/{len(stocks)})</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error processing {stock}: {e}")
                progress_bar.progress(processed_count/len(stocks))

        progress_bar.empty()
        status_text.empty()

    if not assets:
        st.error("❌ Quantum Systems Offline - Market Data Unavailable")
        return

    # Premium Visualization Section
    st.markdown("## 🌠 Market Performance Matrix")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(
            pd.DataFrame(assets),
            x='symbol',
            y='expected_return',
            color='expected_return',
            color_continuous_scale='Viridis',
            template='plotly_dark',
            title="<b>Expected Returns</b>",
            labels={'expected_return': 'Return %', 'symbol': 'Asset'}
        )
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(
            pd.DataFrame(assets),
            x='symbol',
            y='volatility',
            color='volatility',
            color_continuous_scale='Plasma',
            template='plotly_dark',
            title="<b>Risk Volatility</b>",
            labels={'volatility': 'Volatility %', 'symbol': 'Asset'}
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Premium Sentiment Analysis Display
    st.markdown("## 🧠 Neural Market Pulse")
    sentiment_score = analyze_market_sentiment_from_csv(num_rows=200)
    
    sentiment_color = "#4ade80" if sentiment_score >= 0 else "#f87171"
    st.markdown(f"""
    <div class="metric-card" style="border-left: 6px solid {sentiment_color};">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h3 style="margin:0; color: {sentiment_color};">Market Sentiment Cortex</h3>
            <div style="font-size: 1.2rem;">{"🚀 Bullish Momentum" if sentiment_score >=0 else "⚠️ Bearish Warning"}</div>
        </div>
        <div style="display: flex; align-items: baseline; gap: 1rem; margin: 1rem 0;">
            <div style="font-size: 2.5rem; font-weight: bold; color: {sentiment_color};">{sentiment_score:.2f}</div>
            <div style="width: 60%; height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px;">
                <div style="width: {abs(sentiment_score)*50}%; height: 100%; background: {sentiment_color}; 
                    border-radius: 4px; transition: width 1s ease;"></div>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
            <div style="padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 12px;">
                <div style="color: #94a3b8;">Confidence Index</div>
                <div style="font-size: 1.2rem; color: {sentiment_color};">{(abs(sentiment_score)*100):.1f}%</div>
            </div>
            <div style="padding: 1rem; background: rgba(255,255,255,0.05); border-radius: 12px;">
                <div style="color: #94a3b8;">Market Temperature</div>
                <div style="font-size: 1.2rem; color: {sentiment_color};">{'38.6°C' if sentiment_score >=0 else '12.4°C'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quantum Optimization
    with st.spinner("⚛️ Processing Quantum Annealing Sequence..."):
        Q = build_qubo_model(assets, k=k, sentiment_adjustment=sentiment_score * 0.01)
        best_solution = solve_qubo(Q, num_reads=200)
        selected_symbols = [assets[i]['symbol'] for i in best_solution if best_solution[i] == 1]

    st.success("✅ Quantum Optimization Complete - Optimal Portfolio Generated")

    # Premium Results Display
    st.markdown("## 💎 Optimal Quantum Portfolio")
    selected_assets_details = [asset for asset in assets if asset['symbol'] in selected_symbols]
    
    if selected_assets_details:
        weights, allocated_amounts = allocate_funds(selected_assets_details, investment_amount)
        
        # Animated Pie Chart
        fig3 = px.pie(
            names=[a['symbol'] for a in selected_assets_details],
            values=weights,
            hole=0.4,
            title="Portfolio Quantum Distribution",
            color_discrete_sequence=px.colors.sequential.Viridis,
            template='plotly_dark'
        )
        fig3.update_traces(
            textposition='inside',
            textinfo='percent+label',
            marker=dict(line=dict(color='#000000', width=2))
        )
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Premium Allocation Table
        allocation_df = pd.DataFrame({
            "Asset": [asset['symbol'] for asset in selected_assets_details],
            "Weight (%)": (weights * 100).round(2),
            "Allocation": [f"${amt:,.2f}" for amt in allocated_amounts],
            "Risk-Return Ratio": [f"{(asset['expected_return']/asset['volatility']):.2f}" 
                                for asset in selected_assets_details]
        })
        
        st.markdown("""
        <style>
            .dataframe {
                background: rgba(255,255,255,0.05) !important;
                border-radius: 12px !important;
            }
            .dataframe th {
                background: rgba(99, 102, 241, 0.2) !important;
            }
            .dataframe td {
                border-bottom: 1px solid rgba(255,255,255,0.1) !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            allocation_df.style.format({"Weight (%)": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True
        )

    # Raw Data Section
    with st.expander("🔍 Quantum Data Matrix", expanded=False):
        st.dataframe(
            pd.DataFrame(assets).sort_values('expected_return', ascending=False),
            use_container_width=True
        )

if __name__ == "_main_":
    main()
