import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ========================
# CONFIG
# ========================
SEQ_LENGTH = 60
FEATURE_COLUMNS = ['open', 'high', 'low', 'close', 'volume']  # 👈 Multi-feature input
TARGET_COLUMN = 'close'  # 👈 What you want to predict
EPOCHS = 10
BATCH_SIZE = 32
FILE_PATH = "../datasets/BTC-USD_1h_Training data_1739260800000_to_1738425540000.csv"

# ========================
# LOAD AND PREPROCESS
# ========================
def load_data(filepath: str, features: list[str]) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df[features]

def normalize_data(df: pd.DataFrame) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    return scaled, scaler

def create_sequences(data: np.ndarray, target_index: int, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i][target_index])  # 👈 Predict only the target (e.g., close)
    return np.array(X), np.array(y)

# ========================
# MODEL SETUP
# ========================
def build_lstm_model(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        LSTM(64),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ========================
# PLOTTING
# ========================
def plot_predictions(actual: np.ndarray, predicted: np.ndarray):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted')
    plt.title("LSTM OHLCV Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

# ========================
# MAIN PIPELINE
# ========================
def run_pipeline():
    df = load_data(FILE_PATH, FEATURE_COLUMNS)
    scaled_data, scaler = normalize_data(df)

    target_index = FEATURE_COLUMNS.index(TARGET_COLUMN)
    X, y = create_sequences(scaled_data, target_index, SEQ_LENGTH)

    model = build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

    predictions = model.predict(X)

    # Only inverse-transform the predicted target
    close_scaler = MinMaxScaler()
    close_scaler.min_, close_scaler.scale_ = scaler.min_[target_index], scaler.scale_[target_index]
    predicted_prices = close_scaler.inverse_transform(predictions)
    actual_prices = close_scaler.inverse_transform(y.reshape(-1, 1))

    plot_predictions(actual_prices, predicted_prices)

# ========================
# RUN
# ========================
if __name__ == "__main__":
    run_pipeline()
