from abc import ABC, abstractmethod
import pandas as pd
from .base import Strategy  
from ..strategies.marketRegimeStrategy_simplified import MarketRegimeStrategy
from ..strategies.deepPredictorStrategy import DeepPredictorStrategy
from ..models.lg import LogisticRegressionModel 

class MetaFusionStrategy(Strategy):
    def __init__(self, training_dataset_filepath):
        super().__init__(training_dataset_filepath, None)
        self.marketRegimeStrategy = MarketRegimeStrategy(training_dataset_filepath, bullish_threshold=0.7, bearish_threshold=0.3)
        self.deepPredictorStrategy = DeepPredictorStrategy(training_dataset_filepath, seq_length=10, epochs=50, batch_size=32)
        self.meta_model = None
        self.meta_model = None
        self.signals = []
        self.merged_training_df = None
        self.merged_training_filepath = None
        self.merged_predict_df = None
        self.merged_predict_filepath = None

    def train_meta_model(self):
        self.merged_training_df = self.preprocess_data(self.training_dataset_filepath)
        self.merged_training_filepath = self.merged_training_df.to_csv("merged_training_data.csv", index=False)
        self.meta_model = LogisticRegressionModel(self.merged_training_df)
        self.meta_model.train()

    def preprocess_data(self, data_filepath):
        self.set_predict_dataset_filepath(data_filepath)
        marketRegimeData = self.marketRegimeStrategy.generate_signals(self.predict_dataset_filepath)
        deepPredictorData = self.deepPredictorStrategy.generate_signals(self.predict_dataset_filepath)
        concatenated_data = self.merge_model_outputs(marketRegimeData, deepPredictorData)
        return concatenated_data

    def merge_model_outputs(self, marketRegimeData, deepPredictorData):
        lstm_df = deepPredictorData.rename(columns={"predictions": "deepPredictor"})
        hmm_df = marketRegimeData.rename(columns={"predictions": "marketRegime"})

        lstm_df = lstm_df.sort_values(by="timestamp").reset_index(drop=True)
        hmm_df = hmm_df.sort_values(by="timestamp").reset_index(drop=True)

        merged_df = pd.merge(lstm_df, hmm_df, on="timestamp", how="inner")
        merged_df = merged_df.dropna()

        return merged_df

    def generate_signals(self):
        """
        Use the trained meta-model to generate final trading signals.
        """
        self.reset_signals()
        self.merged_predict_df = self.preprocess_data(self.predict_dataset_filepath)
        self.merged_predict_filepath = self.merged_predict_df.to_csv("merged_predict_data.csv", index=False)

        print("Predict data frame in MetaFusionStrategy:", self.merged_predict_df)
        for i in range (len(self.merged_predict_df)):
            print("Predict data frame in MetaFusionStrategy:", self.merged_predict_df.iloc[[i]])
            predicted_decision = self.meta_model.predict(self.merged_predict_df.iloc[[i]])
            self.signals.append(predicted_decision)

        print("Signals in MetaFusionStrategy:", self.signals)
        return self.merged_predict_df

    def execute_trade(self, i, data_row, cash, position, entry_price, entry_index, holding_period, trading_fees, max_holding_period):
        signal = self.signals[i]
        price = data_row['close']
        trade_action = 'hold'
        if signal == 1 or holding_period >= max_holding_period :
            if cash > price * (1 + trading_fees):
                position = cash // (price * (1 + trading_fees))
                cost = position * price * (1 + trading_fees)
                cash -= cost
                entry_price = price
                entry_index = i
                holding_period = 0
                trade_action = 'buy'
        elif signal == -1 or holding_period >= max_holding_period:
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
    
    def buy(self, data_row, cash, position, entry_price, entry_index):
        entry_price = data_row["close"]
        position = cash / entry_price
        cash = 0
        return cash, position, entry_price, entry_index, 0

    def sell(self, data_row, cash, position, entry_price, entry_index):
        cash = position * data_row["close"]
        position = 0
        return cash, position, entry_price, entry_index, 0
    
    def set_thresholds(self, *args, **kwargs):
        return super().set_thresholds(*args, **kwargs)
