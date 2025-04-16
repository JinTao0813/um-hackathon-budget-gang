from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .base import Strategy  
from ..models.hmm_simplified import HMM_simplified
from ..models.lstm import LSTMModel

class MetaFusionStrategy(Strategy):
    def __init__(self, training_dataset_filepath, predict_dataset_filepath=None):
        super().__init__(training_dataset_filepath, predict_dataset_filepath)

        # Initialize models
        self.hmmodel = HMM_simplified(training_dataset_filepath)
        self.lstmmodel = LSTMModel(training_dataset_filepath)

        # Train models
        self.hmmodel.train()
        self.lstmmodel.train()

        # Initialize meta-model (Logistic Regression)
        self.meta_model = LogisticRegression()

        # Preprocessing of data
        self.hmm_data, self.lstm_data = self.preprocess_data(self.training_df)

        # Train the meta-model (stacking approach)
        self.train_meta_model()

    def preprocess_data(self, data):
        """
        Preprocess the data for both HMM and LSTM models.
        """
        # Preprocess data for HMM
        hmm_data = self.hmmodel.preprocess_data(data)
        
        # Preprocess data for LSTM
        lstm_data = self.lstmmodel.preprocess_data(data)
        
        return hmm_data, lstm_data

    def set_thresholds(self, *args, **kwargs):
        """
        Set thresholds for strategy (e.g., for signals like buy, sell).
        """
        self.buy_threshold = kwargs.get("buy_threshold", 0.7)
        self.sell_threshold = kwargs.get("sell_threshold", 0.3)

    def generate_signals(self):
        """
        Generate trading signals based on the prediction data from HMM and LSTM models.
        We combine these signals using a meta-model (Logistic Regression).
        """
        # Get predictions from both models
        hmm_predictions = self.hmmodel.predict(self.predict_df)
        lstm_predictions = self.lstmmodel.predict(self.predict_df)

        # Stack the predictions to use as input for the meta-model
        stacked_predictions = pd.DataFrame({
            "hmm": hmm_predictions,
            "lstm": lstm_predictions
        })

        # Predict using the meta-model
        signals = self.meta_model.predict(stacked_predictions)
        
        return signals

    def train_meta_model(self):
        """
        Train the meta-model using the outputs of HMM and LSTM.
        The output is the optimal rule for combining both models' predictions.
        """
        # Stack predictions from HMM and LSTM to train meta-model
        stacked_train_data = pd.DataFrame({
            "hmm": self.hmm_data["predictions"],
            "lstm": self.lstm_data["predictions"]
        })

        # Use the target labels for training the meta-model (this should be the actual trading signals/labels)
        self.meta_model.fit(stacked_train_data, self.training_df["target_labels"])

    def execute_trade(self, i, data_row, cash, position, entry_price, entry_index, holding_period, trading_fees, max_holding_period):
        """
        Executes a trade decision (buy/sell/hold) based on generated signals.
        """
        # Get the signal at index i from the generated signals
        signal = self.signals[i]
        
        if signal == 1:  # 'buy' signal
            # Execute buy logic
            cash, position, entry_price, entry_index, holding_period = self.buy(data_row, cash, position, entry_price, entry_index)
        elif signal == -1:  # 'sell' signal
            # Execute sell logic
            cash, position, entry_price, entry_index, holding_period = self.sell(data_row, cash, position, entry_price, entry_index)
        else:  # 'hold' signal
            # Hold position, no changes to cash or position
            pass
        
        return cash, position, entry_price, entry_index, holding_period

    def buy(self, data_row, cash, position, entry_price, entry_index):
        # Buy logic implementation
        entry_price = data_row["close"]  # Example
        position = cash / entry_price  # Assume buying full position
        cash = 0  # Cash used up for buying
        return cash, position, entry_price, entry_index, 0

    def sell(self, data_row, cash, position, entry_price, entry_index):
        # Sell logic implementation
        cash = position * data_row["close"]  # Sell the position at current price
        position = 0  # Position sold
        return cash, position, entry_price, entry_index, 0
