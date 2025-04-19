from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    def __init__(self, training_dataset_filepath=None, predict_dataset_filepath=None):
        self.training_dataset_filepath = training_dataset_filepath
        self.predict_dataset_filepath = predict_dataset_filepath
        self.predict_df = None
        if predict_dataset_filepath is not None:
            self.predict_df = pd.read_csv(predict_dataset_filepath)
        self.training_df = pd.read_csv(training_dataset_filepath) if training_dataset_filepath else None
        self.signals = []


    def set_predict_dataset_filepath(self, predict_dataset_filepath):
        """
        Set the prediction dataset file path.
        """
        self.predict_dataset_filepath = predict_dataset_filepath
        self.set_predict_df(pd.read_csv(predict_dataset_filepath))
        
    def set_predict_df(self, predict_df):
        """
        Set the prediction data file path.
        """
        self.predict_df = predict_df


    def reset_signals(self):
        """
        Reset the signals list to an empty state.
        """
        self.signals = []

    @abstractmethod
    def set_thresholds(self, *args, **kwargs):
        """
        Set thresholds for the strategy. This is optional and may not be implemented in all strategies.
        """
        pass

    @abstractmethod
    def generate_signals(self):
        """
        Generate trading signals based on the prediction data.
        """
        pass
