import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os
import random

# Long Short-Term Memory (LSTM) model for time series prediction
class LSTMModel():
    def __init__(self, training_data_filepath:str, seed):
        self.seed = seed # random seed for reproducibility
        self.seq_length = 60 # number of time steps to look back
        self.feature_columns = ['close', 'netflow_total', 'exchange_whale_ratio', 'funding_rates', 'sa_average_dormancy']
        self.target_columns = ['close']  # 👈 What you want to predict
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

    # Normalize the data using MinMaxScaler
    def normalize_data(self, df: pd.DataFrame, scaler: MinMaxScaler = None):
        if scaler is None: # if no scaler is provided, create a new one
            scaler = MinMaxScaler() # Initialize the scaler
            scaled = scaler.fit_transform(df.values) # Fit the scaler to the data and transform it
        else:
            scaled = scaler.transform(df.values) # Transform the data using the existing scaler
        return scaled, scaler # Return the scaled data and the scaler

    # Create sequences of data for LSTM input
    def create_sequences(self, data: np.ndarray, target_indices: list[int], seq_length: int):
        X, y = [], []
        for i in range(seq_length, len(data)):
            X.append(data[i - seq_length:i]) # append the last seq_length data points
            y.append(data[i][target_indices]) # append the target data point
        return np.array(X), np.array(y) 
    
    # Set random seed for reproducibility
    def set_seed(self, seed):
        os.environ['PYTHONHASHSEED'] = str(seed) # Set hash seed for Python
        random.seed(seed) # Set random seed for Python
        np.random.seed(seed) # Set random seed for NumPy
        tf.random.set_seed(seed)

    # Build the LSTM model
    def build_lstm_model(self, input_shape: tuple[int, int], output_dim: int) -> Sequential: 
        self.set_seed(self.seed) # Set random seed for reproducibility
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape), # LSTM layer with 64 units
            LSTM(64),
            Dense(output_dim)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error') # Compile the model with Adam optimizer and mean squared error loss
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
        # Load and normalize training data 
        df_train = self.load_data(self.training_file_path, self.feature_columns)

        scaled_train, self.scaler = self.normalize_data(df_train) # Normalize the training data
        self.target_indices = [self.feature_columns.index(col) for col in self.target_columns] # Get the indices of the target columns
        X_train, y_train = self.create_sequences(scaled_train, self.target_indices, self.seq_length) # Create sequences for training data

        print(f"✅ Training set shape: X={X_train.shape}, y={y_train.shape}")

        # Build and train model
        model = self.build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=len(self.target_columns))
        print("X train ", X_train)
        print("y train ", y_train)
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size)

        self.model = model

    def predict(self, testing_file_path):
        if self.model is None: # if the model is not trained, raise an exception
            raise Exception("Model not trained. Please run the train() method first.")
        # Load and normalize test data 
        df_test = self.load_data(testing_file_path, self.feature_columns, include_time=True) # Load test data with timestamp
        time_labels = df_test['timestamp'].iloc[self.seq_length:]  # Align time with prediction steps

        df_test_no_time = df_test[self.feature_columns]  # drop datetime for normalization
        scaled_test, _ = self.normalize_data(df_test_no_time, self.scaler)
        X_test, y_test = self.create_sequences(scaled_test, self.target_indices, self.seq_length) # Create sequences for test data

        print(f"✅ Testing set shape: X={X_test.shape}, y={y_test.shape}")
        print(f"🕒 Predicting for {len(X_test)} hours ≈ {len(X_test) / 24:.1f} days")

        # Predict and inverse transform 
        predictions = self.model.predict(X_test) # Predict using the model
       

        dummy = np.zeros((predictions.shape[0], len(self.feature_columns))) # create a dummy array to hold the predictions
        dummy[:, self.target_indices] = predictions
        predicted_prices = self.scaler.inverse_transform(dummy)[:, self.target_indices]

        dummy[:, self.target_indices] = y_test
        actual_prices = self.scaler.inverse_transform(dummy)[:, self.target_indices]

        # Save results to CSV 
        self.result_df = pd.DataFrame({
            'timestamp': time_labels,
            **{f'actual_{col}': actual_prices[:, i] for i, col in enumerate(self.target_columns)},
            **{f'predicted_{col}': predicted_prices[:, i] for i, col in enumerate(self.target_columns)},
        })

    
