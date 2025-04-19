from .base import Strategy
from .. models.lstm import LSTMModel

class DeepPredictorStrategy(Strategy):
    def __init__(self, training_dataset_filepath, seq_length, epochs, batch_size, seed):
        super().__init__(training_dataset_filepath, None)
        self.seed = seed
        self.seq_length = seq_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lstm_model = LSTMModel(training_dataset_filepath, self.seed)
        self.lstm_model.train()
        self.result_df = None

    def generate_signals(self, predict_data_filepath, save_file_name):
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
            if curr_pred < prev_close:
                signals.append('buy')
            elif curr_pred > prev_close:
                signals.append('sell')
            else:
                signals.append('hold')

        # Align signals with timestamps (skip the first which has no prev value)
        self.result_df = self.result_df.iloc[1:].copy()
        self.result_df['deepPredictor'] = signals
        self.signals = signals  # Store for use in execute_trade

        self.result_df.to_csv(save_file_name)
        return self.result_df

    
    def set_thresholds(self, *args, **kwargs):
        return super().set_thresholds(*args, **kwargs)