from .base import Strategy
from ..models.hmm_model import HMM
import pandas as pd

class HMMStrategy(Strategy):
    def __init__(self, training_dataset_filepath, predict_dataset_filepath, bullish_threshold, bearish_threshold):
        super().__init__(training_dataset_filepath, predict_dataset_filepath)
        self.holding_period = 0
        self.hmmodel = HMM(training_dataset_filepath)
        self.hmmodel.train()
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold

    def generate_signals(self):
        self.reset_signals()
        print(len(self.predict_df))
        processed_df = self.hmmodel.preprocess_data(self.predict_df)
        
        for i in range(len(processed_df) - 1):
            predicted_probs = self.hmmodel.predict_probabilities(processed_df, i)  
            prob_bullish = predicted_probs.get('bullish', 0)
            prob_bearish = predicted_probs.get('bearish', 0)

            if prob_bullish >= self.bullish_threshold:
                self.signals.append('bullish')
            elif prob_bearish >= self.bearish_threshold:
                self.signals.append('bearish')
            else:
                self.signals.append('neautral')
        return processed_df


    def execute_trade(self, i, data_row, cash, position, entry_price, entry_index, holding_period, trading_fees, max_holding_period):
        price = data_row['close']
        signal = self.signals[i] if i < len(self.signals) else None
        trade_action = 'neautral'

        if signal == 'bullish' and position == 0 and cash > price * (1 + trading_fees):
            position = cash // (price * (1 + trading_fees))
            cost = position * price * (1 + trading_fees)
            cash -= cost
            entry_price = price
            entry_index = i
            holding_period = 0
            trade_action = 'buy'

        elif signal == 'bearish' or holding_period >= max_holding_period:
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
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold

