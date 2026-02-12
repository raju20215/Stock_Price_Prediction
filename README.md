# 📈 NSE Hybrid AI Stock Price Predictor

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A cutting-edge **hybrid AI-powered stock price prediction system** that combines **LSTM deep learning models** with **FinBERT sentiment analysis** to forecast next-day prices for NSE (National Stock Exchange) stocks. The application features a beautiful interactive dashboard built with Streamlit.

![Demo](https://img.shields.io/badge/status-active-success.svg)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Approach & Methodology](#-approach--methodology)
- [Skills Demonstrated](#-skills-demonstrated)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Supported Stocks](#-supported-stocks)
- [Model Architecture](#-model-architecture)
- [Performance Metrics](#-performance-metrics)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Problem Statement

**Challenge**: Traditional stock price prediction models rely solely on historical price data, ignoring critical market sentiment from news and social media. This leads to:

- ❌ Limited prediction accuracy during volatile market conditions
- ❌ Inability to capture sudden market shifts due to news events
- ❌ Slow model switching when analyzing multiple stocks
- ❌ Poor user experience with complex financial analytics tools

**Solution**: This project addresses these challenges by:

- ✅ Combining technical analysis (LSTM) with sentiment analysis (FinBERT)
- ✅ Preloading all models for instant stock switching
- ✅ Providing real-time news sentiment scoring
- ✅ Offering an intuitive, interactive web dashboard

---

## 🎯 Objectives

### Primary Objectives

1. **Accurate Price Prediction**: Develop LSTM models that predict next-day closing prices with high accuracy
2. **Sentiment Integration**: Incorporate real-time news sentiment to adjust predictions dynamically
3. **User Experience**: Create an intuitive web interface for non-technical users
4. **Performance Optimization**: Enable instant switching between 12 different stock models

### Secondary Objectives

1. Visualize model performance with predicted vs. actual price comparisons
2. Display accuracy metrics (MAE, MAPE) for transparency
3. Fetch news from multiple reliable sources with automatic fallback
4. Support high-volatility and low-volatility stock patterns

---

## ✨ Key Features

### 🤖 AI & Machine Learning

- **LSTM Deep Learning Models**: 12 pre-trained models for different NSE stocks
- **FinBERT Sentiment Analysis**: Financial news sentiment scoring using transformer models
- **Hybrid Predictions**: Combines technical indicators + sentiment (±1.5% adjustment)
- **Model Caching**: All models preloaded at startup for instant access

### 📊 Interactive Dashboard

- **Real-time Price Charts**: Predicted vs. Actual price visualization
- **Sentiment Indicators**: Bullish/Bearish/Neutral sentiment scoring
- **Performance Metrics**: MAE and MAPE accuracy display
- **Dual News Sources**: Google News RSS + Yahoo Finance with automatic fallback

### ⚡ Performance & UX

- **Instant Stock Switching**: <1ms model retrieval from cache
- **Responsive UI**: Dark-themed Plotly charts with interactive hover
- **Progressive Loading**: Visual feedback during model initialization
- **Error Handling**: Graceful fallbacks for news API failures

---

## 🛠️ Technology Stack

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.12+ | Primary programming language |
| **TensorFlow** | 2.x | Deep learning framework |
| **Keras** | 3.x | Neural network API |
| **Streamlit** | 1.x | Web application framework |

### Machine Learning & NLP

| Library | Purpose |
|---------|---------|
| **LSTM (Keras)** | Time series prediction |
| **FinBERT** | Financial sentiment analysis |
| **Transformers** | Pre-trained NLP models |
| **scikit-learn** | Data preprocessing & scaling |

### Data & Visualization

| Library | Purpose |
|---------|---------|
| **yfinance** | Yahoo Finance data fetching |
| **pandas** | Data manipulation |
| **numpy** | Numerical computations |
| **Plotly** | Interactive charts |
| **pandas-ta** | Technical indicators |

### Web & Networking

| Library | Purpose |
|---------|---------|
| **requests** | HTTP requests for news APIs |
| **xml.etree** | RSS feed parsing |
| **Beautiful Soup** | HTML parsing (via yfinance) |

---

## 🔬 Approach & Methodology

### 1. Data Collection & Preprocessing

```python
# Historical data from Yahoo Finance (1 year)
Data Features:
├── Close Price
├── Volume
├── Daily Return (calculated)
├── Volatility (20-day rolling std)
├── Price Range (High - Low)
└── Volume Moving Average (20-day)
```

**Preprocessing Steps**:
1. Download 1-year historical data via yfinance
2. Calculate technical indicators (returns, volatility, etc.)
3. Normalize features using MinMaxScaler (0-1 range)
4. Create sliding windows (60 or 90 days based on stock volatility)

### 2. LSTM Model Architecture

```
Input Layer (6 features)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (64 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 3 (32 units)
    ↓
Dense Layer (1 unit) → Next-day Price Prediction
```

**Training Parameters**:
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Metrics: Mean Absolute Error (MAE)
- Window Size: 60 days (low volatility) / 90 days (high volatility)

### 3. Sentiment Analysis Pipeline

```
News Fetching → Headline Extraction → FinBERT Tokenization → Sentiment Scoring
```

**Multi-Source News Fetching**:
1. **Primary**: Google News RSS (India-focused, NSE stocks)
2. **Fallback**: Yahoo Finance API
3. **Processing**: Top 5 headlines → FinBERT → Sentiment score (-1 to +1)

**FinBERT Output**:
- Positive probability
- Negative probability
- Neutral probability
- **Final Score**: (Positive - Negative)

### 4. Hybrid Prediction Fusion

```python
base_prediction = LSTM_model.predict(technical_features)
sentiment_adjustment = sentiment_score * 0.015  # Max ±1.5%
final_prediction = base_prediction * (1 + sentiment_adjustment)
```

### 5. Performance Evaluation

**Metrics Calculated**:
- **MAE** (Mean Absolute Error): Average prediction error in ₹
- **MAPE** (Mean Absolute Percentage Error): Percentage accuracy
- **Visual Comparison**: 60-day predicted vs. actual overlay

---

## 💡 Skills Demonstrated

### 🧠 Machine Learning & AI

- [x] **Deep Learning**: LSTM architecture design and training
- [x] **Time Series Forecasting**: Sequential data prediction
- [x] **Transfer Learning**: Using pre-trained FinBERT models
- [x] **Model Optimization**: Hyperparameter tuning for stock volatility
- [x] **Ensemble Methods**: Combining LSTM + sentiment analysis

### 💻 Software Engineering

- [x] **Full-Stack Development**: Backend (Python) + Frontend (Streamlit)
- [x] **Performance Optimization**: Model caching, lazy loading
- [x] **API Integration**: Multiple news sources with fallback logic
- [x] **Error Handling**: Graceful degradation for API failures
- [x] **Code Organization**: Modular functions, clean architecture

### 📊 Data Science

- [x] **Data Preprocessing**: Feature engineering, normalization
- [x] **Technical Analysis**: Volatility, returns, moving averages
- [x] **Statistical Analysis**: MAE, MAPE metrics
- [x] **Data Visualization**: Interactive Plotly charts
- [x] **Web Scraping**: RSS feed parsing, HTML extraction

### 🔧 DevOps & Tools

- [x] **Version Control**: Git/GitHub
- [x] **Dependency Management**: requirements.txt
- [x] **Environment Configuration**: Python virtual environments
- [x] **Documentation**: Comprehensive README, code comments

---

## 🚀 Installation

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- 4GB+ RAM (for model loading)
- Internet connection (for news fetching)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/nse-stock-predictor.git
cd nse-stock-predictor
```

### Step 2: Create Virtual Environment (Optional but Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow; print(tensorflow.__version__)"
```

---

## 🎮 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Interface Guide

#### 1. **Stock Selection**
- Use the sidebar dropdown to select from 12 NSE stocks
- Models are pre-cached, switching is instant!

#### 2. **Main Dashboard**
- **Left Panel**: Predicted vs. Actual price comparison chart (60 days)
- **Right Panel**: 
  - Sentiment score with news headlines
  - Model accuracy metrics (MAE, MAPE)

#### 3. **Price Forecasting**
- Click "🚀 Run Hybrid Forecast" button
- View next-day price prediction
- See the hybrid fusion of LSTM + sentiment

#### 4. **Sentiment Analysis**
- Toggle "Enable Sentiment Analysis" checkbox
- View news source being used (Google News / Yahoo Finance)
- Expand headlines to read full titles

### Example Workflow

1. Select **RELIANCE.NS** from sidebar
2. View historical predicted vs. actual performance
3. Check sentiment score (e.g., 0.42 BULLISH)
4. Click "Run Hybrid Forecast"
5. Get next-day prediction: ₹2,850.50 (+1.2%)

---

## 📁 Project Structure

```
nse-stock-predictor/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── NEWS_CONFIG.md         # News sources documentation
│
├── models/                # Pre-trained LSTM models
│   ├── RELIANCE.NS_model.h5
│   ├── TCS.NS_model.h5
│   ├── HDFCBANK.NS_model.h5
│   ├── INFY.NS_model.h5
│   ├── LICI.NS_model.h5
│   ├── HINDUNILVR.NS_model.h5
│   ├── ITC.NS_model.h5
│   ├── BAJFINANCE.NS_model.h5
│   ├── AXISBANK.NS_model.h5
│   ├── ADANIENT.NS_model.h5
│   ├── KOTAKBANK.NS_model.h5
│   └── MARUTI.NS_model.h5
│
├── scalers/               # MinMaxScaler objects
│   ├── RELIANCE.NS_scaler.pkl
│   ├── TCS.NS_scaler.pkl
│   └── ... (12 scalers total)
│
└── results/               # Training results (optional)
    └── training_summary.csv
```

---

## 📈 Supported Stocks

| Ticker | Company Name | Sector | Volatility |
|--------|-------------|--------|------------|
| **RELIANCE.NS** | Reliance Industries | Energy/Retail | Low |
| **TCS.NS** | Tata Consultancy Services | IT Services | Low |
| **HDFCBANK.NS** | HDFC Bank | Banking | Low |
| **INFY.NS** | Infosys | IT Services | Low |
| **LICI.NS** | Life Insurance Corporation | Insurance | Low |
| **HINDUNILVR.NS** | Hindustan Unilever | FMCG | Low |
| **ITC.NS** | ITC Limited | FMCG/Hotels | Low |
| **BAJFINANCE.NS** | Bajaj Finance | NBFC | Low |
| **AXISBANK.NS** | Axis Bank | Banking | Low |
| **ADANIENT.NS** | Adani Enterprises | Conglomerate | **High** |
| **KOTAKBANK.NS** | Kotak Mahindra Bank | Banking | Low |
| **MARUTI.NS** | Maruti Suzuki | Automotive | **High** |

**Note**: High volatility stocks use 90-day windows; low volatility stocks use 60-day windows.

---

## 🏗️ Model Architecture

### LSTM Network Details

```python
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
lstm_1 (LSTM)              (None, 60, 128)           69,120    
dropout_1 (Dropout)        (None, 60, 128)           0         
lstm_2 (LSTM)              (None, 60, 64)            49,408    
dropout_2 (Dropout)        (None, 60, 64)            0         
lstm_3 (LSTM)              (None, 32)                12,416    
dense (Dense)              (None, 1)                 33        
=================================================================
Total params: 130,977
Trainable params: 130,977
Non-trainable params: 0
```

### FinBERT Model

- **Base Model**: `yiyanghkust/finbert-tone`
- **Architecture**: BERT transformer (110M parameters)
- **Fine-tuned on**: Financial news and analyst reports
- **Output**: 3-class sentiment (positive, negative, neutral)

---

## 📊 Performance Metrics

### LSTM Model Performance

Typical accuracy across different stocks:

| Stock | MAE (₹) | MAPE | Accuracy |
|-------|---------|------|----------|
| RELIANCE.NS | ₹45-60 | 2.1% | ~97.9% |
| TCS.NS | ₹70-90 | 2.3% | ~97.7% |
| HDFCBANK.NS | ₹30-45 | 2.0% | ~98.0% |
| INFY.NS | ₹25-35 | 1.8% | ~98.2% |
| MARUTI.NS | ₹180-220 | 2.5% | ~97.5% |

**Note**: Performance varies with market conditions and volatility.

### Sentiment Analysis Impact

- **Bullish News** (+0.3 to +1.0): Increases prediction by 0.5% to 1.5%
- **Neutral News** (-0.1 to +0.1): Minimal adjustment (<0.2%)
- **Bearish News** (-1.0 to -0.3): Decreases prediction by 0.5% to 1.5%

---

## 🔮 Future Enhancements

### Short-term Roadmap

- [ ] Add more NSE stocks (expand to top 50)
- [ ] Multi-day predictions (3-day, 7-day forecasts)
- [ ] Export predictions to CSV/Excel
- [ ] Email alerts for significant price movements
- [ ] Dark/Light theme toggle

### Long-term Vision

- [ ] **Advanced Models**: 
  - Transformer-based models (Temporal Fusion Transformers)
  - GRU and Bidirectional LSTM comparisons
  - Ensemble voting from multiple models

- [ ] **Enhanced Sentiment**:
  - Twitter sentiment analysis
  - Reddit r/IndianStreetBets integration
  - Corporate announcement tracking

- [ ] **Portfolio Management**:
  - Multi-stock portfolio tracking
  - Risk analysis and diversification suggestions
  - Backtesting capabilities

- [ ] **Mobile App**:
  - React Native/Flutter mobile version
  - Push notifications
  - Watchlist management

- [ ] **API Development**:
  - REST API for predictions
  - Webhook integrations
  - Third-party app support

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Reporting Issues

1. Check if the issue already exists
2. Open a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected vs. actual behavior
   - Screenshots (if applicable)

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints where possible
- Write meaningful commit messages

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

### Libraries & Frameworks

- **TensorFlow/Keras Team** - Deep learning framework
- **Hugging Face** - FinBERT pre-trained model
- **Streamlit** - Beautiful web app framework
- **yfinance** - Yahoo Finance data access

### Data Sources

- **Yahoo Finance** - Historical stock data
- **Google News** - Real-time news headlines
- **NSE India** - National Stock Exchange data

### Inspiration

This project was inspired by the need for accessible AI-powered financial tools for retail investors in India.

---

## 📞 Contact & Support

- **Author**: [Your Name]
- **Email**: your.email@example.com
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn](https://linkedin.com/in/yourprofile)

### Getting Help

1. Check the [Issues](https://github.com/yourusername/nse-stock-predictor/issues) page
2. Read [NEWS_CONFIG.md](NEWS_CONFIG.md) for news setup
3. Open a new issue with the `question` label

---

## ⭐ Show Your Support

If this project helped you, please give it a ⭐ on GitHub!

---

## 📝 Changelog

### Version 1.0.0 (Current)
- ✅ 12 pre-trained LSTM models for NSE stocks
- ✅ FinBERT sentiment analysis integration
- ✅ Google News RSS + Yahoo Finance dual sources
- ✅ Interactive Streamlit dashboard
- ✅ Model caching for instant switching
- ✅ Predicted vs. Actual visualization
- ✅ Real-time accuracy metrics

---

<div align="center">

**Built with ❤️ for the Indian Stock Market**

Made by [Your Name] | 2026

</div>
