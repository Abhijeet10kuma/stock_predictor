
# Live Stock Predictor

This is a Streamlit-based advanced stock prediction web app that uses an LSTM model to forecast the next day's closing stock price.

## Features
- Enter any stock symbol (e.g., AAPL, TSLA)
- Uses Yahoo Finance to fetch historical data
- Trains an LSTM neural network in real-time
- Predicts next-day closing price
- Plotly interactive graph

## Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
streamlit run stock_predictor_app.py
```

### 3. To Use the Windows `.exe` (placeholder)
Navigate to `windows/` and double-click `stock_predictor.exe` (will be replaced with real app later).

## Requirements
- streamlit
- yfinance
- pandas, numpy
- plotly
- scikit-learn
- tensorflow

## Roadmap
- [ ] Transformer-based forecasting (Coming soon)
- [ ] Real-time graph streaming
- [ ] Portfolio optimization module
