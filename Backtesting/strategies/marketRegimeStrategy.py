from .base import Strategy
from ..models.hmm import HmmModel
import pandas as pd

# MarketRegimeStrategy class use Hidden Markov Model (HMM) to predict market regimes
class MarketRegimeStrategy(Strategy):
    def __init__(self, training_dataset_filepath, seed, bullish_threshold, bearish_threshold):
        super().__init__(training_dataset_filepath, None)
        self.seed = seed
        self.bullish_threshold = bullish_threshold # threshold for bullish signal
        self.bearish_threshold = bearish_threshold # threshold for bearish signal
        self.hmmodel = HmmModel(training_dataset_filepath, seed) # initialize HMM model
        self.hmmodel.train()
        self.result_df = None

    def generate_signals(self, predict_data_filepath, save_file_name): # predict_data_filepath: str
        self.set_predict_dataset_filepath(predict_data_filepath) # set the prediction dataset file path
        self.reset_signals()
        processed_df = self.hmmodel.preprocess_data(self.predict_df) # preprocess the data

        prob_bullish_list = [] # list to store bullish probabilities
        prob_bearish_list = [] # list to store bearish probabilities
        signal_list = [] # list to store signals

        for i in range(len(processed_df)): # iterate through the processed DataFrame
            predicted_probs = self.hmmodel.predict_probabilities(processed_df, i) # get predicted probabilities
            prob_bullish = predicted_probs.get('buy', 0) # get bullish probability ( 0 indicating to get the state label )
            prob_bearish = predicted_probs.get('sell', 0) 

            prob_bullish_list.append(prob_bullish) # append bullish probability to the list
            prob_bearish_list.append(prob_bearish)

            if prob_bullish >= self.bullish_threshold: # # check if bullish probability exceeds threshold
                signal_list.append('buy')
            elif prob_bearish >= self.bearish_threshold: # check if bearish probability exceeds threshold
                signal_list.append('sell')
            else:
                signal_list.append('hold')

        # Add columns to the original processed_df
        processed_df['prob_bullish'] = prob_bullish_list
        processed_df['prob_bearish'] = prob_bearish_list
        processed_df['marketRegime'] = signal_list

        self.result_df = processed_df # store the result DataFrame
        self.result_df.to_csv(save_file_name, index=False) # save the result DataFrame to a CSV file
        return self.result_df
    
    def set_thresholds(self, bullish_threshold, bearish_threshold): # set the thresholds for heatmap backtesting
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold

