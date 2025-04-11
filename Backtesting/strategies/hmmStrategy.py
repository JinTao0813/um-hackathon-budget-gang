from .base import Strategy
from ..models.hmm_model import HMM
import pandas as pd


class HMMStrategy(Strategy):
    def __init__(self, training_dataset_filepath, predict_dataset_filepath):
        super().__init__(training_dataset_filepath, predict_dataset_filepath)
        self.holding_period = 0
        self.hmmodel = HMM(training_dataset_filepath)
        self.hmmodel.train()
        self.predict_df= pd.read_csv(predict_dataset_filepath)

    def generate_signals(self):
        processed_df = self.hmmodel.preprocess_data(self.predict_df)
        for i in range(len(processed_df)-1):
            print("\nCurrent index: ", i)
            predicted_state = self.hmmodel.predict(processed_df, i)
            self.signals.append(predicted_state)

