import pandas as pd
from .base import Strategy  
from .marketRegimeStrategy import MarketRegimeStrategy
from ..strategies.deepPredictorStrategy import DeepPredictorStrategy
from ..models.lg import LogisticRegressionModel 
import numpy as np
import random

class MetaFusionStrategy(Strategy):
    def __init__(self, hmm_dataset_filepath, lstm_dataset_filepath, seed=42):
        self.seed = seed
        self.set_seed(self.seed)
        super().__init__(None, None)
        self.hmm_dataset_filepath = hmm_dataset_filepath
        self.lstm_dataset_filepath = lstm_dataset_filepath
        self.hmm_predict_dataset_filepath = hmm_dataset_filepath
        self.lstm_predict_dataset_filepath = lstm_dataset_filepath
        self.marketRegimeStrategy = MarketRegimeStrategy(self.hmm_dataset_filepath, self.seed, bullish_threshold=0.6, bearish_threshold=0.3)
        self.deepPredictorStrategy = DeepPredictorStrategy(self.lstm_dataset_filepath, seq_length=10, epochs=50, batch_size=32, seed=self.seed)
        self.signals = []
        self.merged_training_df = None
        self.merged_training_filepath = None
        self.merged_predict_df = None
        self.merged_predict_filepath = None

    def set_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def preprocess_data(self, hmm_dataset_filepath, lstm_dataset_filepath):
        marketRegimeData = self.marketRegimeStrategy.generate_signals(hmm_dataset_filepath)
        deepPredictorData = self.deepPredictorStrategy.generate_signals(lstm_dataset_filepath)
        concatenated_df = self.merge_model_outputs(marketRegimeData, deepPredictorData)
        return concatenated_df
    
    def get_merged_predict_df(self, hmm_predict_dataset_filepath, lstm_predict_dataset_filepath):
        self.merged_predict_df = self.preprocess_data(hmm_predict_dataset_filepath, lstm_predict_dataset_filepath)
        self.merged_predict_filepath = self.merged_predict_df.to_csv("merged_predict_data.csv", index=False)

    def merge_model_outputs(self, marketRegimeData, deepPredictorData):
        lstm_df = deepPredictorData.rename(columns={"predictions": "deepPredictor"})
        hmm_df = marketRegimeData.rename(columns={"predictions": "marketRegime"})

        lstm_df = lstm_df.sort_values(by="timestamp").reset_index(drop=True)
        hmm_df = hmm_df.sort_values(by="timestamp").reset_index(drop=True)

        merged_df = pd.merge(lstm_df, hmm_df, on="timestamp", how="inner")
        merged_df = merged_df.dropna()
        merged_df = merged_df.drop_duplicates()

        return merged_df

    def weighted_vote(self, df, weight_hmm=0.4, weight_lstm=0.6):
        votes = {}
        votes[df['marketRegime']] = votes.get(df['marketRegime'], 0) + weight_hmm
        votes[df['deepPredictor']] = votes.get(df['deepPredictor'], 0) + weight_lstm
        print("Votes:", votes)
        return max(votes, key=votes.get)

    def generate_signals(self):
        self.merged_predict_df = self.preprocess_data(self.hmm_predict_dataset_filepath, self.lstm_predict_dataset_filepath)
        self.reset_signals()

        # Apply weighted vote
        self.merged_predict_df['ensemble_prediction'] = self.merged_predict_df.apply(self.weighted_vote, axis=1)

        
        # Convert predictions to signal list
        self.signals = self.merged_predict_df['ensemble_prediction'].tolist()

        print("Signals in MetaFusionStrategy:", self.signals)
        return self.merged_predict_df


    def execute_trade(self, i, data_row, cash, position, entry_price, entry_index, holding_period, trading_fees, max_holding_period):
        price = data_row['close']
        signal = self.signals[i] if i < len(self.signals) else None
        trade_action = 'neautral'

        if signal == 'buy' and position == 0 and cash > price * (1 + trading_fees):
            position = cash // (price * (1 + trading_fees))
            cost = position * price * (1 + trading_fees)
            cash -= cost
            entry_price = price
            entry_index = i
            holding_period = 0
            trade_action = 'buy'

        elif signal == 'sell' or holding_period >= max_holding_period:
            if entry_index is not None and position > 0:
                sell_value = position * price * (1 - trading_fees)
                cash += sell_value
                position = 0
                entry_price = 0
                entry_index = None
                holding_period = 0
                trade_action = 'sell'
        else:
            holding_period += 1

        return cash, position, entry_price, entry_index, holding_period, trade_action
        
    def set_thresholds(self, bullish_threshold, bearish_threshold):
        self.marketRegimeStrategy.set_thresholds(bullish_threshold, bearish_threshold)

    def set_predict_dataset_filepath(self, hmm_predict_dataset_filepath, lstm_predict_dataset_filepath):
        """
        Set the prediction dataset file path.
        """
        self.hmm_predict_dataset_filepath = hmm_predict_dataset_filepath
        self.lstm_predict_dataset_filepath = lstm_predict_dataset_filepath

    