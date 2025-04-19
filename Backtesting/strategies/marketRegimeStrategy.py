from .base import Strategy
from ..models.hmm import HmmModel
import pandas as pd

class MarketRegimeStrategy(Strategy):
    def __init__(self, training_dataset_filepath, seed, bullish_threshold, bearish_threshold):
        super().__init__(training_dataset_filepath, None)
        self.seed = seed
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.hmmodel = HmmModel(training_dataset_filepath, seed)
        self.hmmodel.train()
        self.result_df = None

    def generate_signals(self, predict_data_filepath, save_file_name):
        self.set_predict_dataset_filepath(predict_data_filepath)
        self.reset_signals()
        # print("Predict data fram in marketregimestrategy:", self.predict_df)
        processed_df = self.hmmodel.preprocess_data(self.predict_df)

        prob_bullish_list = []
        prob_bearish_list = []
        signal_list = []

        for i in range(len(processed_df)):
            predicted_probs = self.hmmodel.predict_probabilities(processed_df, i)
            prob_bullish = predicted_probs.get('buy', 0)
            prob_bearish = predicted_probs.get('sell', 0)

            prob_bullish_list.append(prob_bullish)
            prob_bearish_list.append(prob_bearish)

            if prob_bullish >= self.bullish_threshold:
                signal_list.append('buy')
            elif prob_bearish >= self.bearish_threshold:
                signal_list.append('sell')
            else:
                signal_list.append('hold')

        # Add columns to the original processed_df
        processed_df['prob_bullish'] = prob_bullish_list
        processed_df['prob_bearish'] = prob_bearish_list
        processed_df['marketRegime'] = signal_list

        self.result_df = processed_df
        self.result_df.to_csv(save_file_name, index=False)
        return self.result_df
    
    def set_thresholds(self, bullish_threshold, bearish_threshold):
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold

