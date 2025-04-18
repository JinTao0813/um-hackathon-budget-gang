import pandas as pd
from .base import Strategy  
from .marketRegimeStrategy import MarketRegimeStrategy
from ..strategies.deepPredictorStrategy import DeepPredictorStrategy
from ..models.lg import LogisticRegressionModel 

class MetaFusionStrategy(Strategy):
    def __init__(self, hmm_dataset_filepath, lstm_dataset_filepath):
        super().__init__(None, None)
        self.hmm_dataset_filepath = hmm_dataset_filepath
        self.lstm_dataset_filepath = lstm_dataset_filepath
        self.hmm_predict_dataset_filepath = hmm_dataset_filepath
        self.lstm_predict_dataset_filepath = lstm_dataset_filepath
        self.marketRegimeStrategy = MarketRegimeStrategy(self.hmm_dataset_filepath, bullish_threshold=0.7, bearish_threshold=0.3)
        self.deepPredictorStrategy = DeepPredictorStrategy(self.lstm_dataset_filepath, seq_length=10, epochs=50, batch_size=32)
        self.meta_model = None
        self.meta_model = None
        self.signals = []
        self.merged_training_df = None
        self.merged_training_filepath = None
        self.merged_predict_df = None
        self.merged_predict_filepath = None

    def train_meta_model(self, hmm_dataset_filepath, lstm_dataset_filepath):
        self.merged_training_df = self.preprocess_data(hmm_dataset_filepath, lstm_dataset_filepath)
        self.merged_training_filepath = self.merged_training_df.to_csv("merged_training_data.csv", index=False)
        self.meta_model = LogisticRegressionModel(self.merged_training_df)
        self.meta_model.train()

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

    def generate_signals(self):
        self.merged_predict_df = self.preprocess_data(self.hmm_predict_dataset_filepath, self.lstm_predict_dataset_filepath)
        self.reset_signals()

        print("Predict data frame in MetaFusionStrategy:", self.merged_predict_df)
        for i in range (len(self.merged_predict_df)):
            print("Predict data frame in MetaFusionStrategy:", self.merged_predict_df.iloc[[i]])
            predicted_decision = self.meta_model.predict(self.merged_predict_df.iloc[[i]])
            self.signals.append(predicted_decision)

        print("Signals in MetaFusionStrategy:", self.signals)
        return self.merged_predict_df

    def execute_trade(self, i, data_row, cash, position, entry_price, entry_index, holding_period, trading_fees, max_holding_period):
        print("Signals in execute_trade:", self.signals)
        signal = self.signals[i]
        print("Signal in an execute_trade:", signal)
        price = data_row['close']
        print("Price in an execute_trade:", price)
        trade_action = 'hold'
        if signal == 'buy' or holding_period >= max_holding_period :
            print("1")
            if cash > price * (1 + trading_fees):
                print("2")
                position = cash // (price * (1 + trading_fees))
                cost = position * price * (1 + trading_fees)
                cash -= cost
                entry_price = price
                entry_index = i
                holding_period = 0
                trade_action = 'buy'
        elif signal == 'sell' or holding_period >= max_holding_period:
            print("3")
            if entry_index is not None and position > 0:
                print("4")
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
    
    def set_thresholds(self, bullish_threshold, bearish_threshold):
        self.marketRegimeStrategy.set_thresholds(bullish_threshold, bearish_threshold)

    def set_predict_dataset_filepath(self, hmm_predict_dataset_filepath, lstm_predict_dataset_filepath):
        """
        Set the prediction dataset file path.
        """
        self.hmm_predict_dataset_filepath = hmm_predict_dataset_filepath
        self.lstm_predict_dataset_filepath = lstm_predict_dataset_filepath