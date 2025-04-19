import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import os
import joblib
from ..utils.indicator import IndicatorCalculator
from ..utils.featureNormalizer import ExtrernalNormalizer, SelfNormalizer
import matplotlib.pyplot as plt

# Hidden Markov Model (HMM) for market state prediction
class HmmModel():
    def __init__(self, training_data_filepath, seed=42):
        self.seed = seed # random seed for reproducibility
        self.model_path = 'models/hmm.pkl' # path to save the model
        self.model = None
        self.training_data = pd.read_csv(training_data_filepath) # load training data
        self.df = self.preprocess_data(self.training_data) # preprocess the data
        self.market_state_labels = {} # dictionary to store market state labels
        self.stats = None
        self.initialize_metrics() # calculate initial metrics
        self.features = ['log_return', 'exchange_whale_ratio', 'netflow_total'] # features for HMM

    def preprocess_data(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp']) 
        df = df.sort_values('timestamp')
        df = df.dropna()

        ind_calc = (
            IndicatorCalculator(df)
            .add_returns()
        )
        result_df = ind_calc.get_df() # add log returns
        return result_df.dropna() # drop rows with NaN values

    def initialize_metrics(self):
        print(self.df)
        metrics = ['netflow_total', 'exchange_whale_ratio', 'funding_rates', 'sa_average_dormancy']
        self.stats = self.df[metrics].agg(['mean', 'std']).T # calculate mean and std for each metric


    def train(self):
        features_to_normalize = ['netflow_total', 'exchange_whale_ratio', 'funding_rates', 'sa_average_dormancy'] # features to normalize

        normalizer = SelfNormalizer(self.df) # initialize normalizer
        normalized_features = normalizer.normalize(features_to_normalize) # normalize the features
    
        observations = self.extract_observations(normalized_features, self.features) # extract observations
        
        self.model = self.train_hmm_model(observations) # train HMM model with the observations
        df = self.assign_states(self.df, self.model, self.features) # assign states to the data
        
        df, state_labels = self.identify_market_states(df) # identify market states
        
        self.df = df # update the dataframe with assigned states
        self.market_state_labels = state_labels # store the market state labels
        
        print("Converged:", self.model.monitor_.converged)
        print("Final log likelihood:", self.model.monitor_.history[-1])
        print("State Labels:", self.market_state_labels)
        print(df[['timestamp', 'open', 'close', 'netflow_total', 'exchange_whale_ratio', 'funding_rates', 'sa_average_dormancy', 'log_return', 'state', 'market_state']])
        
        self.save_model()


    def extract_observations(self, df, observation_columns): # extract observations from the dataframe
        return df[observation_columns].values


    def train_hmm_model(self, observations, n_states=3, n_iter=1000): # train HMM model
        # Ensure observations are in the correct shape
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=n_iter, verbose=True, random_state=self.seed)
        model.fit(observations)
        return model


    def assign_states(self, df, model, observation_columns): # assign states to the data
        observations = self.extract_observations(df, observation_columns) # extract observations
        df['state'] = model.predict(observations) # predict the hidden states
        return df


    def identify_market_states(self, df): # identify market states based on the log returns
        print("Unique states:", df['state'].unique()) 
        state_analysis = df.groupby('state').agg({'log_return': 'mean'})
        bullish = state_analysis['log_return'].idxmax() # state with max log return
        bearish = state_analysis['log_return'].idxmin() # state with min log return
        neutral = list(set(df['state'].unique()) - {bullish, bearish})[0] # state with neutral log return
        
        label_map = {
            int(bullish): 'buy',  # state with max log return
            int(bearish): 'sell', # state with min log return
            int(neutral): 'hold' # state with neutral log return
        }

        print(df['state'].unique()) 
        df['market_state'] = df['state'].map(label_map) # map states to labels
        self.plot_market_states(df,'close')   # plot market states for visualisation
        return df, label_map

    def save_model(self): # save the trained model
        os.makedirs("models", exist_ok=True) # Create the directory if it doesn't exist
        joblib.dump(self.model, self.model_path) # save the model

    def load_model(self, filepath): # load the trained model
        if os.path.exists(filepath): # check if the model file exists
            model = joblib.load(filepath)
            print("Model loaded successfully.")
            return model
        else:
            print("Model file not found.")
            return None
        
    def predict(self, predict_df, today_index): # predict the market state for a given index

        today_df = predict_df.iloc[today_index] # get today's data

        # Get yesterday's data
        if today_index == 0: # If the first index, use the first two rows
            yesterday_df = predict_df.iloc[today_index] 
        elif today_index == len(predict_df) - 1: # If the last index, use the last two rows
            yesterday_df = predict_df.iloc[today_index - 2] # use the second last row
        else: # Otherwise, use the previous row
            yesterday_df = predict_df.iloc[today_index - 1] # get yesterday's data
    
        print("\nToday index: ", today_index)
        print(type(today_df))
        print(today_df)

        normalizer = ExtrernalNormalizer(stats = self.stats, full_df=self.df) # initialize normalizer
        features = normalizer.get_all_features(today_df, yesterday_df) # get features for today and yesterday from the normalizer

        new_obs = np.array([list(features.values())]) # convert to numpy array

        # Predict hidden state
        state_today = self.model.predict(new_obs)
        
        market_state_today = self.market_state_labels[state_today[0]] # map state to label

        return market_state_today

    def predict_probabilities(self, df, i): # predict the probabilities of each state for a given index
        posterior_probs = self.model.predict_proba(df[self.features].values) # get the probabilities of each state
        
        # Get the probabilities for time step i
        if i < len(posterior_probs):
            probs = posterior_probs[i] # probabilities for the current time step
        else:
            probs = posterior_probs[-1] # probabilities for the last time step

        # Map state index to labels
        return {
            self.market_state_labels.get(idx, f'state_{idx}'): prob
            for idx, prob in enumerate(probs)
        }
    
    def plot_market_states(self, df, price_column):
        # Ensure market_state is a category for consistent coloring
        df['market_state'] = df['market_state'].astype('category')

        # Create color map
        color_map = {'buy': 'green', 'sell': 'red', 'hold': 'orange'}
        colors = df['market_state'].map(color_map)

        # Plot
        plt.figure(figsize=(14, 6))
        plt.scatter(df['timestamp'], df[price_column], c=colors, label='Market State', s=10)
        plt.plot(df['timestamp'], df[price_column], color='gray', alpha=0.3, label='Price Trend')
        
        # Legend handling
        import matplotlib.patches as mpatches
        legend_handles = [mpatches.Patch(color=color_map[state], label=state.capitalize()) for state in color_map]
        plt.legend(handles=legend_handles)

        plt.title('Market States Over Time')
        plt.xlabel('Time')
        plt.ylabel(price_column.capitalize())
        plt.grid(True)
        plt.tight_layout()
        plt.show()

