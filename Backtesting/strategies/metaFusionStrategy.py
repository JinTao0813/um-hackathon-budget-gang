from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .base import Strategy  
from ..strategies.marketRegimeStrategy_simplified import MarketRegimeStrategy
from ..strategies.deepPredictorStrategy import DeepPredictorStrategy

class MetaFusionStrategy(Strategy):
    def __init__(self, training_dataset_filepath):
        super().__init__(training_dataset_filepath, None)
        self.marketRegimeStrategy = MarketRegimeStrategy(training_dataset_filepath, bullish_threshold=0.7, bearish_threshold=0.3)
        self.deepPredictorStrategy = DeepPredictorStrategy(training_dataset_filepath, seq_length=10, epochs=50, batch_size=32)
        self.marketRegimeData = None
        self.deepPredictorData = None
        self.concatenated_data = None

    def preprocess_data(self, predict_data_filepath):
        self.set_predict_dataset_filepath(predict_data_filepath)
        self.marketRegimeData = self.marketRegimeStrategy.generate_signals(self.predict_dataset_filepath)
        self.deepPredictorData = self.deepPredictorStrategy.generate_signals(self.predict_dataset_filepath)
    
        self.concatenated_data = self.merge_model_outputs()
        print("Merged DataFrame:")
        print(self.concatenated_data)
    
    def merge_model_outputs(self):
        lstm_df = self.deepPredictorData
        hmm_df = self.marketRegimeData
        on = 'timestamp'
        how = 'inner'
        # Ensure the 'on' column exists in both DataFrames
        if on not in lstm_df.columns or on not in hmm_df.columns:
            raise ValueError(f"'{on}' column must exist in both DataFrames.")

        # Optional: sort both by timestamp to make it clean
        lstm_df = lstm_df.sort_values(by=on).reset_index(drop=True)
        hmm_df = hmm_df.sort_values(by=on).reset_index(drop=True)

        # Merge the DataFrames
        merged_df = pd.merge(lstm_df, hmm_df, on=on, how=how, suffixes=('_lstm', '_hmm'))

        return merged_df


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
