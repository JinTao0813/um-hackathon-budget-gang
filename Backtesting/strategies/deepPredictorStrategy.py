from .base import Strategy
from .. models.lstm import LSTMModel
import pandas as pd


class DeepPredictorStrategy(Strategy):
    def __init__(self, training_dataset_filepath, seq_length, epochs, batch_size):
        super().__init__(training_dataset_filepath, None)
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_model = LSTMModel(training_dataset_filepath)
        self.lstm_model.train()
        self.result_df = None

    def generate_signals(self, predict_data_filepath):
        print("Predict data filepath in DeepPredictorStrategy:", predict_data_filepath)
        self.set_predict_dataset_filepath(predict_data_filepath)
        print("Predict data filepath in DeepPredictorStrategy:", self.predict_dataset_filepath)
        if self.lstm_model is None:
            raise ValueError("LSTM model not provided.")

        # Run predictions
        self.lstm_model.predict(self.predict_dataset_filepath)
        self.result_df = self.lstm_model.result_df

        print("Results from LSTM model:")
        print(self.result_df)

        signals = []
        predicted_close = self.result_df['predicted_close']
        actual_close = self.result_df['actual_close']

        for i in range(1, len(predicted_close)):
            prev_close = actual_close.iloc[i - 1]
            curr_pred = predicted_close.iloc[i]

            # Simple logic: predict ↑ => buy, predict ↓ => sell
            if curr_pred > prev_close:
                signals.append('buy')
            elif curr_pred < prev_close:
                signals.append('sell')
            else:
                signals.append('hold')

        # Align signals with timestamps (skip the first which has no prev value)
        self.result_df = self.result_df.iloc[1:].copy()
        self.result_df['deepPredictor'] = signals
        self.signals = signals  # Store for use in execute_trade

        print("✅ Signals generated using LSTM predictions")
        return self.result_df

    def execute_trade(self, i, data_row, cash, position, entry_price, entry_index, holding_period, trading_fees, max_holding_period):
        signal = self.result_df.iloc[i]['signal']
        close_price = data_row['close']

        if signal == 'buy' and position == 0:
            position = 1
            entry_price = close_price
            entry_index = i
            cash -= close_price + trading_fees
            trade_signal = 'buy'
        elif signal == 'sell' and position == 1:
            position = 0
            cash += close_price - trading_fees
            trade_signal = 'sell'
        else:
            trade_signal = 'hold'

        return cash, position, entry_price, entry_index, holding_period + 1, trade_signal   
    
    def set_thresholds(self, *args, **kwargs):
        return super().set_thresholds(*args, **kwargs)