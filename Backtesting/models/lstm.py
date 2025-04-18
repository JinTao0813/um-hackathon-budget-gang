import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import random

class LSTMModel():
    def __init__(self, training_data_filepath:str, seed):
        self.seed = seed
        self.seq_length = 60 # number of time steps to look back
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume']  # 👈 Multi-feature input
        self.target_columns = ['close', 'high', 'low']  # 👈 What you want to predict
        self.epochs = 10 # number of times the entire dataset will be passed through the model during training
        self.batch_size = 32 # how many samples the model processes before updating its weights during training.
        self.training_file_path = training_data_filepath
        self.model = None
        self.target_indices = None
        self.scaler = None
        self.predict_df = None
        self.result_df = None

    def load_data(seld, filepath: str, features: list[str], include_time=False) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        if include_time:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df[['timestamp'] + features]
        return df[features]

    def normalize_data(self, df: pd.DataFrame, scaler: MinMaxScaler = None):
        if scaler is None:
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(df.values)
        else:
            scaled = scaler.transform(df.values)
        return scaled, scaler

    def create_sequences(self, data: np.ndarray, target_indices: list[int], seq_length: int):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i])
            y.append(data[i][target_indices])
        return np.array(X), np.array(y)
    
    def set_seed(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def build_lstm_model(self, input_shape: tuple[int, int], output_dim: int) -> Sequential:
        self.set_seed(self.seed)
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            LSTM(64),
            Dense(output_dim)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def plot_predictions(self, actual: np.ndarray, predicted: np.ndarray, columns: list[str], time_labels=None):
        plt.figure(figsize=(15, 6))
        for i, col in enumerate(columns):
            plt.subplot(1, len(columns), i + 1)
            x = time_labels if time_labels is not None else np.arange(len(actual))
            plt.plot(x, actual[:, i], label='Actual')
            plt.plot(x, predicted[:, i], label='Predicted')
            plt.title(f"{col.capitalize()} Prediction")
            plt.xlabel("Time")
            plt.ylabel(col.capitalize())
            plt.legend()
        plt.tight_layout()
        plt.show()


    def train(self):
        # === Load and normalize training data ===
        df_train = self.load_data(self.training_file_path, self.feature_columns)
        scaled_train, self.scaler = self.normalize_data(df_train)

        self.target_indices = [self.feature_columns.index(col) for col in self.target_columns]
        X_train, y_train = self.create_sequences(scaled_train, self.target_indices, self.seq_length)

        print(f"✅ Training set shape: X={X_train.shape}, y={y_train.shape}")

        # === Build and train model ===
        model = self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=len(self.target_columns))
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

        self.model = model

    def predict(self, testing_file_path):
        if self.model is None:
            raise Exception("Model not trained. Please run the train() method first.")
        # === Load and normalize test data ===
        df_test = self.load_data(testing_file_path, self.feature_columns, include_time=True)
        time_labels = df_test['timestamp'].iloc[self.seq_length:]  # Align time with prediction steps

        df_test_no_time = df_test[self.feature_columns]  # drop datetime for normalization
        scaled_test, _ = self.normalize_data(df_test_no_time, self.scaler)
        X_test, y_test = self.create_sequences(scaled_test, self.target_indices, self.seq_length)

        print(f"✅ Testing set shape: X={X_test.shape}, y={y_test.shape}")
        print(f"🕒 Predicting for {len(X_test)} hours ≈ {len(X_test) / 24:.1f} days")

        # === Predict and inverse transform ===
        predictions = self.model.predict(X_test)
       

        dummy = np.zeros((predictions.shape[0], len(self.feature_columns)))
        dummy[:, self.target_indices] = predictions
        predicted_prices = self.scaler.inverse_transform(dummy)[:, self.target_indices]

        dummy[:, self.target_indices] = y_test
        actual_prices = self.scaler.inverse_transform(dummy)[:, self.target_indices]

        # === Plot results ===
        self.plot_predictions(actual_prices, predicted_prices, self.target_columns, time_labels)

        # === Save results to CSV ===
        self.result_df = pd.DataFrame({
            'timestamp': time_labels,
            **{f'actual_{col}': actual_prices[:, i] for i, col in enumerate(self.target_columns)},
            **{f'predicted_{col}': predicted_prices[:, i] for i, col in enumerate(self.target_columns)},
        })
        self.result_df.to_csv("prediction_results.csv", index=False)
        print("📁 Results saved to prediction_results.csv")

    
