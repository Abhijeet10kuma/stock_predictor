
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Function to load and preprocess data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    df = data[['Close']]
    return df

def create_dataset(data, seq_len=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(seq_len, len(scaled_data)):
        X.append(scaled_data[i - seq_len:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(60, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(60))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_stock(data, prediction=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data['Close'], name='Actual Close', mode='lines'))
    if prediction is not None:
        fig.add_trace(go.Scatter(y=[None]*(len(data)-1) + [prediction], name='Predicted Next', mode='lines+markers'))
    st.plotly_chart(fig)

# Streamlit UI
st.set_page_config(page_title="Live Stock Predictor", layout="wide")
st.title("📈 Advanced Live Stock Market Predictor (LSTM-Based)")

ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA):", "AAPL").upper()
start_date = st.date_input("Start Date", datetime.date(2015, 1, 1))
end_date = datetime.datetime.today().date()

if st.button("Predict with LSTM"):
    df = load_data(ticker, start_date, end_date)
    if df.empty:
        st.error("No data found. Please check ticker symbol or date range.")
    else:
        st.success(f"Fetched {len(df)} rows for {ticker}")
        X, y, scaler = create_dataset(df)
        model = build_lstm_model((X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        last_60_days = df[-60:].values
        last_scaled = scaler.transform(last_60_days)
        X_test = np.reshape(last_scaled, (1, 60, 1))
        prediction = model.predict(X_test)
        predicted_price = scaler.inverse_transform(prediction)[0][0]

        st.subheader("Predicted Close Price for Next Day:")
        st.metric(label="📊 Prediction", value=f"${predicted_price:.2f}")

        st.subheader("Stock Price Graph")
        plot_stock(df, predicted_price)

# Future Transformer Model Section Placeholder
if st.checkbox("Enable Transformer-based Model (Coming Soon)"):
    st.info("Transformer-based prediction will be available in the next version.")
