import pandas as pd
from .base import Strategy  
from .marketRegimeStrategy import MarketRegimeStrategy
from .deepPredictorStrategy import DeepPredictorStrategy
import numpy as np
import random
import matplotlib.pyplot as plt

class FlowSignalStrategy(Strategy):
    def __init__(self, hmm_dataset_filepath, lstm_dataset_filepath, seed=42):
        self.seed = seed
        self.set_seed(self.seed)
        super().__init__(None, None)
        self.hmm_dataset_filepath = hmm_dataset_filepath
        self.lstm_dataset_filepath = lstm_dataset_filepath
        self.hmm_predict_dataset_filepath = hmm_dataset_filepath
        self.lstm_predict_dataset_filepath = lstm_dataset_filepath
        self.marketRegimeStrategy = MarketRegimeStrategy(self.hmm_dataset_filepath, self.seed, bullish_threshold=0.6, bearish_threshold=0.3) # Initialize the MarketRegimeStrategy
        self.deepPredictorStrategy = DeepPredictorStrategy(self.lstm_dataset_filepath, seq_length=10, epochs=120, batch_size=32, seed=self.seed) # Initialize the DeepPredictorStrategy
        self.signals = []
        self.merged_training_df = None
        self.merged_training_filepath = None
        self.merged_predict_df = None
        self.merged_predict_filepath = "merged_predict_data.csv"
        self.file_idx = 1

    def set_seed(self, seed): # Set random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

    def preprocess_data(self, hmm_dataset_filepath, lstm_dataset_filepath):
        marketRegimeData = self.marketRegimeStrategy.generate_signals(hmm_dataset_filepath, self.get_file_name()) # Generate signals using MarketRegimeStrategy
        deepPredictorData = self.deepPredictorStrategy.generate_signals(lstm_dataset_filepath, self.get_file_name()) # Generate signals using DeepPredictorStrategy
        concatenated_df = self.merge_model_outputs(marketRegimeData, deepPredictorData) # Merge the outputs of both models
        return concatenated_df 
    
    def merge_model_outputs(self, marketRegimeData, deepPredictorData):
        lstm_df = deepPredictorData.rename(columns={"predictions": "deepPredictor"}) # Rename the predictions column to deepPredictor
        hmm_df = marketRegimeData.rename(columns={"predictions": "marketRegime"}) # Rename the predictions column to marketRegime

        lstm_df = lstm_df.sort_values(by="timestamp").reset_index(drop=True) # Sort the DataFrame by timestamp
        hmm_df = hmm_df.sort_values(by="timestamp").reset_index(drop=True) # Sort the DataFrame by timestamp

        merged_df = pd.merge(lstm_df, hmm_df, on="timestamp", how="inner") # Merge the two DataFrames on timestamp
        merged_df = merged_df.dropna()
        merged_df = merged_df.drop_duplicates()

        return merged_df
        
    def get_merged_predict_df(self, hmm_predict_dataset_filepath, lstm_predict_dataset_filepath): # Get the merged prediction DataFrame
        self.merged_predict_df = self.preprocess_data(hmm_predict_dataset_filepath, lstm_predict_dataset_filepath) # Preprocess the data
        self.merged_predict_df.to_csv(self.get_file_name(), index=False) # Save the merged DataFrame to a CSV file

    # A simple logic to combine signals which is based on the market regime and deep predictor signals
    def combine_signal(self):
        def get_ensemble(row):
            if row['deepPredictor'] == 'buy' and row['marketRegime'] == 'buy':
                return 'buy'
            elif row['deepPredictor'] == 'sell' and row['marketRegime'] == 'sell':
                return 'sell'
            else:
                return row['marketRegime'] if row['marketRegime'] != 'hold' else row['deepPredictor']

        self.merged_predict_df['ensemble_prediction'] = self.merged_predict_df.apply(get_ensemble, axis=1)

    def generate_signals(self): # Generate signals using the combined signals from both models
        self.get_merged_predict_df(self.hmm_predict_dataset_filepath, self.lstm_predict_dataset_filepath)
        self.reset_signals()
        self.combine_signal() # Combine the signals from both models
        self.plot_signal(self.merged_predict_df, 'close') 
        
        self.signals = self.merged_predict_df['ensemble_prediction'].tolist() # Convert the ensemble predictions to a list
        return self.merged_predict_df

    def set_thresholds(self, bullish_threshold, bearish_threshold): # Set the thresholds for the market regime strategy
        self.marketRegimeStrategy.set_thresholds(bullish_threshold, bearish_threshold) # Set the thresholds for the MarketRegimeStrategy

    def set_predict_dataset_filepath(self, hmm_predict_dataset_filepath, lstm_predict_dataset_filepath): # Set the prediction dataset file paths
        """
        Set the prediction dataset file path.
        """
        self.hmm_predict_dataset_filepath = hmm_predict_dataset_filepath
        self.lstm_predict_dataset_filepath = lstm_predict_dataset_filepath

    def get_file_name(self): #  Get the file name for saving the processed data
        file_name = f"processed_data_{self.file_idx}.csv"
        self.file_idx += 1
        return file_name

    def plot_signal(self, df, price_column): # Plot the signals
        # Ensure market_state is a category for consistent coloring
        df[''] = df['ensemble_prediction'].astype('category')

        # Create color map
        color_map = {'buy': 'green', 'sell': 'red', 'hold': 'orange'}
        colors = df['ensemble_prediction'].map(color_map)

        # Plot
        plt.figure(figsize=(14, 6))
        plt.scatter(df['timestamp'], df[price_column], c=colors, label='Final Trade Decision', s=10)
        plt.plot(df['timestamp'], df[price_column], color='gray', alpha=0.3, label='Price Trend')
        
        # Legend handling
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=color_map[state], label=state.capitalize()) for state in color_map]
        plt.legend(handles=legend_handles)

        plt.title('Final Trade Decision Over Time')
        plt.xlabel('Time')
        plt.ylabel(price_column.capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    