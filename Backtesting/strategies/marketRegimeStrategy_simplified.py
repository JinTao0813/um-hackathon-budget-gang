from .base import Strategy
from ..models.hmm_simplified import HMM_simplified
import pandas as pd

class MarketRegimeStrategy(Strategy):
    def __init__(self, training_dataset_filepath, bullish_threshold, bearish_threshold):
        super().__init__(training_dataset_filepath, None)
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
        self.hmmodel = HMM_simplified(training_dataset_filepath)
        self.hmmodel.train()
        self.result_df = None

    def generate_signals(self, predict_data_filepath):
        self.set_predict_dataset_filepath(predict_data_filepath)
        self.reset_signals()
        print("Predict data fram in marketregimestrategy:", self.predict_df)
        processed_df = self.hmmodel.preprocess_data(self.predict_df)

        prob_bullish_list = []
        prob_bearish_list = []
        signal_list = []

        for i in range(len(processed_df)):
            predicted_probs = self.hmmodel.predict_probabilities(processed_df, i)
            prob_bullish = predicted_probs.get('bullish', 0)
            prob_bearish = predicted_probs.get('bearish', 0)

            prob_bullish_list.append(prob_bullish)
            prob_bearish_list.append(prob_bearish)

            if prob_bullish >= self.bullish_threshold:
                signal_list.append('bullish')
            elif prob_bearish >= self.bearish_threshold:
                signal_list.append('bearish')
            else:
                signal_list.append('neutral')

        # Add columns to the original processed_df
        processed_df['prob_bullish'] = prob_bullish_list
        processed_df['prob_bearish'] = prob_bearish_list
        processed_df['marketRegime'] = signal_list

        self.result_df = processed_df
        return self.result_df

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

