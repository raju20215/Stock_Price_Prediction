import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import pandas_ta as ta
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
from datetime import datetime, timedelta
import os
import os
# Force TensorFlow to use the legacy Keras 2 (if needed for your .h5 models)
os.environ['TF_USE_LEGACY_KERAS'] = '1'
# Silence oneDNN and Info logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
# Now your existing imports will work
from tensorflow.keras.models import load_model
import tensorflow as tf
# Use the AutoModel class for better compatibility with TF weights
from transformers import BertTokenizer, TFBertForSequenceClassification

# If the direct import still fails, use this "Hugging Face" recommended fallback:
try:
    from transformers import TFBertForSequenceClassification
except ImportError:
    # This force-loads the TF specific module
    import transformers.models.bert.modeling_tf_bert as tf_bert
    TFBertForSequenceClassification = tf_bert.TFBertForSequenceClassification

st.write("✅ TensorFlow & FinBERT Loaded Successfully")

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="NSE Hybrid AI Predictor", layout="wide", page_icon="📈")

# Stocks from your training summary
SAVED_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'LICI.NS', 
    'HINDUNILVR.NS', 'ITC.NS', 'BAJFINANCE.NS', 'AXISBANK.NS', 
    'ADANIENT.NS', 'KOTAKBANK.NS', 'MARUTI.NS'
]

# Volatility list from your notebook (Determines window size)
HIGH_VOL_STOCKS = ['MARUTI.NS', 'M&M.NS', 'TATAMOTORS.NS', 'TITAN.NS', 
                   'BHARTIARTL.NS', 'LT.NS', 'SUNPHARMA.NS', 'ADANIENT.NS']

@st.cache_resource
def load_all_models():
    """Preload ALL models, scalers at startup and cache them for instant access"""
    st.info("🚀 Initializing AI Engine - Loading all 12 stock models into cache...")
    
    models_dict = {}
    scalers_dict = {}
    
    # Create a progress bar for loading feedback
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(SAVED_STOCKS):
        status_text.text(f"Loading {ticker}... ({idx + 1}/{len(SAVED_STOCKS)})")
        
        # Load LSTM model and scaler for this stock
        model = load_model(f'models/{ticker}_model.h5', compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        scaler = joblib.load(f'scalers/{ticker}_scaler.pkl')
        
        models_dict[ticker] = model
        scalers_dict[ticker] = scaler
        
        # Update progress
        progress_bar.progress((idx + 1) / len(SAVED_STOCKS))
    
    # Load FinBERT once (shared across all stocks)
    status_text.text("Loading FinBERT sentiment model...")
    model_name = 'yiyanghkust/finbert-tone'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    fb_model = TFBertForSequenceClassification.from_pretrained(model_name, from_pt=True)
    
    progress_bar.empty()
    status_text.empty()
    st.success("✅ All models loaded successfully! Ready for predictions.")
    
    return models_dict, scalers_dict, tokenizer, fb_model

# --- 2. SENTIMENT ENGINE (FinBERT TF) WITH MULTIPLE NEWS SOURCES ---

def fetch_news_from_google_rss(ticker):
    """Fetch news from Google News RSS feed (no API key needed)"""
    try:
        import requests
        import xml.etree.ElementTree as ET
        
        # Extract company name from ticker
        company = ticker.replace('.NS', '').replace('.', '+')
        
        url = f"https://news.google.com/rss/search?q={company}+stock&hl=en-IN&gl=IN&ceid=IN:en"
        
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            headlines = []
            
            # Parse RSS feed
            for item in root.findall('.//item')[:10]:
                title = item.find('title')
                if title is not None and title.text:
                    # Clean up the title (Google News adds source at the end)
                    clean_title = title.text.split(' - ')[0]
                    headlines.append(clean_title)
            
            return headlines if headlines else None
        return None
    except Exception as e:
        return None

def fetch_news_from_yfinance(ticker):
    """Original yfinance news fetching"""
    try:
        stock = yf.Ticker(ticker)
        news = None
        
        try:
            news = stock.news
        except:
            pass
        
        if not news:
            try:
                news = stock.get_news()
            except:
                pass
        
        if not news or len(news) == 0:
            return None
        
        headlines = []
        for item in news[:10]:
            title = None
            if isinstance(item, dict):
                title = item.get('title') or item.get('headline')
            else:
                title = getattr(item, 'title', None) or getattr(item, 'headline', None)
            
            if title:
                headlines.append(str(title))
        
        return headlines if headlines else None
    except:
        return None

def get_finbert_sentiment(ticker, tokenizer, fb_model):
    """Fetch headlines from multiple sources and calculate sentiment score using FinBERT"""
    headlines = None
    source_used = "None"
    
    try:
        # Try different news sources in order of preference
        
        # 1. Try Google News RSS (most reliable, no API key needed)
        headlines = fetch_news_from_google_rss(ticker)
        if headlines:
            source_used = "Google News RSS"
        
        # 2. Fallback to yfinance
        if not headlines:
            headlines = fetch_news_from_yfinance(ticker)
            if headlines:
                source_used = "Yahoo Finance"
        
        # If all sources failed
        if not headlines:
            return 0.0, ["⚠️ News sources unavailable. Using neutral sentiment."], "None"
        
        # Take top 5 headlines
        headlines = headlines[:5]
        
        # Tokenize for TensorFlow
        inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="tf", max_length=512)
        
        # Inference
        outputs = fb_model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        
        # Index 1: Positive, Index 2: Negative
        avg_probs = tf.reduce_mean(probs, axis=0)
        sentiment_score = float(avg_probs[1] - avg_probs[2])
        
        return sentiment_score, headlines, source_used
        
    except Exception as e:
        error_msg = str(e)[:100]
        return 0.0, [f"❌ Error: {error_msg}"], "Error"

# --- 3. PRELOAD ALL MODELS AT STARTUP ---
models_dict, scalers_dict, fb_tokenizer, fb_model = load_all_models()

# --- 4. SIDEBAR ---
st.sidebar.title("🇮🇳 NSE Market Pulse")
selected_stock = st.sidebar.selectbox("Select Production Model", SAVED_STOCKS)

# Get the selected stock's model and scaler from cache (instant access!)
model = models_dict[selected_stock]
scaler = scalers_dict[selected_stock]

# --- 5. MAIN DASHBOARD ---
st.title(f"Market Intelligence: {selected_stock}")
col1, col2 = st.columns([2, 1])

with col1:
    df_main = yf.download(selected_stock, period="6mo", progress=False)
    if isinstance(df_main.columns, pd.MultiIndex):
        df_main.columns = df_main.columns.get_level_values(0)
    
    # Prepare features for prediction
    df_main['Daily_Return'] = df_main['Close'].pct_change()
    df_main['Volatility'] = df_main['Close'].rolling(20).std()
    df_main['Price_Range'] = df_main['High'] - df_main['Low']
    df_main['Volume_MA'] = df_main['Volume'].rolling(20).mean()
    df_main.dropna(inplace=True)
    
    win_size = 90 if selected_stock in HIGH_VOL_STOCKS else 60
    features_list = ['Close', 'Volume', 'Daily_Return', 'Volatility', 'Price_Range', 'Volume_MA']
    
    # Generate predictions for the last 30 days to compare with actual
    predictions = []
    actual_dates = []
    
    # Start generating predictions from win_size onwards
    for i in range(win_size, len(df_main)):
        window_data = df_main.iloc[i-win_size:i][features_list]
        scaled_input = scaler.transform(window_data)
        X_input = np.reshape(scaled_input, (1, win_size, 6))
        
        # Predict
        raw_pred_scaled = model.predict(X_input, verbose=0)
        dummy = np.zeros((1, 6)); dummy[0, 0] = raw_pred_scaled[0, 0]
        pred_price = scaler.inverse_transform(dummy)[0, 0]
        
        predictions.append(pred_price)
        actual_dates.append(df_main.index[i])
    
    # Create comparison plot - Last 60 days for better visibility
    compare_days = min(60, len(predictions))
    plot_dates = actual_dates[-compare_days:]
    plot_predictions = predictions[-compare_days:]
    plot_actual = df_main.loc[plot_dates, 'Close'].values
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_dates, y=plot_actual, 
                             name='Actual Price', 
                             line=dict(color='#00D9FF', width=2),
                             mode='lines+markers'))
    fig.add_trace(go.Scatter(x=plot_dates, y=plot_predictions, 
                             name='Predicted Price', 
                             line=dict(color='#FF6B00', width=2, dash='dash'),
                             mode='lines+markers'))
    
    fig.update_layout(
        template="plotly_dark", 
        height=450,
        title="Model Performance: Predicted vs Actual Prices (Last 60 Days)",
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("📰 FinBERT Sentiment")
    
    # Add toggle for sentiment
    use_sentiment = st.checkbox("Enable Sentiment Analysis", value=True, 
                                help="Fetches news from Google News RSS and Yahoo Finance. No API keys required!")
    
    if use_sentiment:
        with st.spinner("Fetching news from multiple sources..."):
            sentiment_score, news_titles, news_source = get_finbert_sentiment(selected_stock, fb_tokenizer, fb_model)
        
        label = "BULLISH" if sentiment_score >= 0.1 else "BEARISH" if sentiment_score <= -0.1 else "NEUTRAL"
        st.metric("Sentiment Score", f"{sentiment_score:.2f}", label)
        
        # Show which source was used
        if news_source != "None" and news_source != "Error":
            st.caption(f"📡 Source: {news_source}")
        
        with st.expander("Recent Headlines", expanded=True):
            for title in news_titles: 
                # Color code the messages
                if "⚠️" in title or "❌" in title:
                    st.warning(title)
                else:
                    st.write(f"• {title}")
    else:
        sentiment_score = 0.0
        st.info("Sentiment analysis disabled. Using neutral sentiment (0.0) for predictions.")
    
    # Model Performance Metrics
    st.divider()
    st.subheader("📊 Model Accuracy")
    
    # Calculate accuracy metrics from the predictions
    mae = np.mean(np.abs(np.array(plot_predictions) - np.array(plot_actual)))
    mape = np.mean(np.abs((np.array(plot_actual) - np.array(plot_predictions)) / np.array(plot_actual))) * 100
    
    st.metric("Mean Absolute Error", f"₹{mae:.2f}")
    st.metric("Accuracy (MAPE)", f"{100 - mape:.2f}%")
    st.caption(f"Based on last {compare_days} predictions")

# --- 6. PREDICTION LOGIC ---
st.divider()
st.subheader("🤖 AI Price Forecasting")

if st.button("🚀 Run Hybrid Forecast"):
    with st.spinner("Processing technical indicators..."):
        df_ml = yf.download(selected_stock, period="1y", progress=False)
        if isinstance(df_ml.columns, pd.MultiIndex):
            df_ml.columns = df_ml.columns.get_level_values(0)
            
        # Features from Notebook: Close, Volume, Daily_Return, Volatility, Price_Range, Volume_MA
        df_ml['Daily_Return'] = df_ml['Close'].pct_change()
        df_ml['Volatility'] = df_ml['Close'].rolling(20).std()
        df_ml['Price_Range'] = df_ml['High'] - df_ml['Low']
        df_ml['Volume_MA'] = df_ml['Volume'].rolling(20).mean()
        df_ml.dropna(inplace=True)
        
        win_size = 90 if selected_stock in HIGH_VOL_STOCKS else 60
        features_list = ['Close', 'Volume', 'Daily_Return', 'Volatility', 'Price_Range', 'Volume_MA']
        
        last_window = df_ml.tail(win_size)[features_list]
        scaled_input = scaler.transform(last_window)
        X_input = np.reshape(scaled_input, (1, win_size, 6))
        
        # LSTM Inference
        raw_pred_scaled = model.predict(X_input, verbose=0)
        
        # Inverse Scaling
        dummy = np.zeros((1, 6)); dummy[0, 0] = raw_pred_scaled[0, 0]
        base_forecast = scaler.inverse_transform(dummy)[0, 0]
        
        # Hybrid Fusion (1.5% max nudge)
        final_forecast = base_forecast * (1 + (sentiment_score * 0.015))
        
        # Visualization
        history = df_ml.tail(15)
        next_day = history.index[-1] + timedelta(days=1)
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=history.index, y=history['Close'], name='History'))
        fig_pred.add_trace(go.Scatter(x=[history.index[-1], next_day], y=[history['Close'].iloc[-1], final_forecast], 
                                     name='Hybrid Forecast', line=dict(color='#00FF00' if final_forecast > history['Close'].iloc[-1] else '#FF0000', width=4)))
        fig_pred.update_layout(template="plotly_dark", title="Next-Day Projection")
        st.plotly_chart(fig_pred, width="stretch")
        
        # Final Stats
        curr_p = df_ml['Close'].iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Current", f"₹{curr_p:.2f}")
        c2.metric("LSTM Baseline", f"₹{base_forecast:.2f}")
        c3.metric("Hybrid Result", f"₹{final_forecast:.2f}", f"{final_forecast - curr_p:.2f}")