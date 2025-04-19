import pandas as pd
from .base import Strategy  
from .marketRegimeStrategy import MarketRegimeStrategy
from .deepPredictorStrategy import DeepPredictorStrategy
import numpy as np
import random

class FlowSignalStrategy(Strategy):
    def __init__(self, hmm_dataset_filepath, lstm_dataset_filepath, seed=42):
        self.seed = seed
        self.set_seed(self.seed)
        super().__init__(None, None)
        self.hmm_dataset_filepath = hmm_dataset_filepath
        self.lstm_dataset_filepath = lstm_dataset_filepath
        self.hmm_predict_dataset_filepath = hmm_dataset_filepath
        self.lstm_predict_dataset_filepath = lstm_dataset_filepath
        self.marketRegimeStrategy = MarketRegimeStrategy(self.hmm_dataset_filepath, self.seed, bullish_threshold=0.6, bearish_threshold=0.3)
        self.deepPredictorStrategy = DeepPredictorStrategy(self.lstm_dataset_filepath, seq_length=10, epochs=120, batch_size=32, seed=self.seed)
        self.signals = []
        self.merged_training_df = None
        self.merged_training_filepath = None
        self.merged_predict_df = None
        self.merged_predict_filepath = "merged_predict_data.csv"
        self.file_idx = 1

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def preprocess_data(self, hmm_dataset_filepath, lstm_dataset_filepath):
        marketRegimeData = self.marketRegimeStrategy.generate_signals(hmm_dataset_filepath, self.get_file_name())
        deepPredictorData = self.deepPredictorStrategy.generate_signals(lstm_dataset_filepath, self.get_file_name())
        concatenated_df = self.merge_model_outputs(marketRegimeData, deepPredictorData)
        return concatenated_df
    
    def merge_model_outputs(self, marketRegimeData, deepPredictorData):
        lstm_df = deepPredictorData.rename(columns={"predictions": "deepPredictor"})
        hmm_df = marketRegimeData.rename(columns={"predictions": "marketRegime"})

        lstm_df = lstm_df.sort_values(by="timestamp").reset_index(drop=True)
        hmm_df = hmm_df.sort_values(by="timestamp").reset_index(drop=True)

        merged_df = pd.merge(lstm_df, hmm_df, on="timestamp", how="inner")
        merged_df = merged_df.dropna()
        merged_df = merged_df.drop_duplicates()

        return merged_df
        
    def get_merged_predict_df(self, hmm_predict_dataset_filepath, lstm_predict_dataset_filepath):
        self.merged_predict_df = self.preprocess_data(hmm_predict_dataset_filepath, lstm_predict_dataset_filepath)
        self.merged_predict_df.to_csv(self.get_file_name(), index=False)


    def weighted_vote(self, df, weight_hmm=0.4, weight_lstm=0.6):
        votes = {}
        votes[df['marketRegime']] = votes.get(df['marketRegime'], 0) + weight_hmm
        votes[df['deepPredictor']] = votes.get(df['deepPredictor'], 0) + weight_lstm
        # print("Votes:", votes)
        return max(votes, key=votes.get)
    
    def combine_signal(self):
        def get_ensemble(row):
            if row['deepPredictor'] == 'buy' and row['marketRegime'] == 'buy':
                return 'buy'
            elif row['deepPredictor'] == 'sell' and row['marketRegime'] == 'sell':
                return 'sell'
            else:
                return row['marketRegime']

        self.merged_predict_df['ensemble_prediction'] = self.merged_predict_df.apply(get_ensemble, axis=1)

    def generate_signals(self):
        self.get_merged_predict_df(self.hmm_predict_dataset_filepath, self.lstm_predict_dataset_filepath)
        self.reset_signals()
        # self.combine_signal()

        # Apply weighted vote
        # self.merged_predict_df['ensemble_prediction'] = self.merged_predict_df.apply(self.weighted_vote, axis=1)
        
        self.merged_predict_df.to_csv(self.get_file_name(), index=False)
        # Convert predictions to signal list
        # self.signals = self.merged_predict_df['ensemble_prediction'].tolist()

        ## Use the marketRegime column as the signal
        # self.signals = self.merged_predict_df['marketRegime'].tolist() 
        # forward result = 0.6486 (bull = buy, bear = sell)
        # forward result = -1.39 (bull = sell, bear = buy)

        # Use the deepPredictor column as the signal
        self.signals = self.merged_predict_df['deepPredictor'].tolist()
        # forward result = 1.18 (too few trade)


        # print("Signals in MetaFusionStrategy:", self.signals)
        return self.merged_predict_df

        
    def set_thresholds(self, bullish_threshold, bearish_threshold):
        self.marketRegimeStrategy.set_thresholds(bullish_threshold, bearish_threshold)

    def set_predict_dataset_filepath(self, hmm_predict_dataset_filepath, lstm_predict_dataset_filepath):
        """
        Set the prediction dataset file path.
        """
        self.hmm_predict_dataset_filepath = hmm_predict_dataset_filepath
        self.lstm_predict_dataset_filepath = lstm_predict_dataset_filepath

    def get_file_name(self):
        file_name = f"processed_data_{self.file_idx}.csv"
        self.file_idx += 1
        return file_name

    