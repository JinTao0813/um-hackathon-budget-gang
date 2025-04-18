import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import os
import joblib
from ..utils.indicator import IndicatorCalculator
from ..utils.featureNormalizer import ExtrernalNormalizer, SelfNormalizer


class HmmModel():
    def __init__(self, training_data_filepath):
        self.model_path = 'models/hmm.pkl'
        self.model = None
        self.training_data = pd.read_csv(training_data_filepath)
        self.df = self.preprocess_data(self.training_data)
        self.market_state_labels = {}
        self.stats = None
        self.initialize_metrics()
        self.features = ['log_return', 'netflow_total', 'exchange_whale_ratio', 'funding_rates', 'sa_average_dormancy']

    def preprocess_data(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df = df.dropna()

        ind_calc = (
            IndicatorCalculator(df)
        #     .add_rsi()
        #     .add_macd()
        #     .add_ema(window=12)
        #     .add_ema(window=26)
        #     .add_sma(window=20)
        #     .add_volatility()
            .add_returns()
        #     .add_candle_stats()
        #     .add_rolling_volatility()
        #     .add_volume_stats()
        #     .add_bollinger_bands()
        )
        result_df = ind_calc.get_df()
        return result_df.dropna()

        # return df

    def initialize_metrics(self):
        print(self.df)
        # metrics = ['volume', 'rsi', 'macd', 'ema_12', 'ema_26', 'sma_20', 'volatility']
        # self.stats = self.df[metrics].agg(['mean', 'std']).T
        metrics = ['netflow_total', 'exchange_whale_ratio', 'funding_rates', 'sa_average_dormancy']
        self.stats = self.df[metrics].agg(['mean', 'std']).T


    def train(self):
        features_to_normalize = [
            'netflow_total', 'exchange_whale_ratio', 'funding_rates', 'sa_average_dormancy']

        # features_to_normalize = [
        #     'volume', 'rsi', 'macd', 'ema_12', 'ema_26', 'sma_20', 'volatility',
        #     'ohlc_mean', 'price_range', 'candle_body', 'direction',
        #     'rolling_volatility', 'volume_ema', 'volume_spike',
        #     'bollinger_upper', 'bollinger_lower', 'bollinger_width'
        # ]

        normalizer = SelfNormalizer(self.df)
        normalized_features = normalizer.normalize(features_to_normalize)
    
        observations = self.extract_observations(normalized_features, self.features)
        
        self.model = self.train_hmm_model(observations)
        df = self.assign_states(self.df, self.model, self.features)
        
        df, state_labels = self.identify_market_states(df)
        
        self.df = df
        self.market_state_labels = state_labels
        
        print("Converged:", self.model.monitor_.converged)
        print("Final log likelihood:", self.model.monitor_.history[-1])
        print("State Labels:", self.market_state_labels)
        print(df[['timestamp', 'open', 'close', 'netflow_total', 'exchange_whale_ratio', 'funding_rates', 'sa_average_dormancy', 'log_return', 'state', 'market_state']])
        
        self.save_model()


    def extract_observations(self, df, observation_columns):
        return df[observation_columns].values


    def train_hmm_model(self, observations, n_states=3, n_iter=1000):
        model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=n_iter, verbose=True)
        model.fit(observations)
        return model


    def assign_states(self, df, model, observation_columns):
        observations = self.extract_observations(df, observation_columns)
        df['state'] = model.predict(observations)
        return df


    def identify_market_states(self, df):
        print("Unique states:", df['state'].unique())
        state_analysis = df.groupby('state').agg({'log_return': 'mean'})
        bullish = state_analysis['log_return'].idxmax()
        bearish = state_analysis['log_return'].idxmin()
        neutral = list(set(df['state'].unique()) - {bullish, bearish})[0]
        
        label_map = {
            int(bullish): 'bullish',
            int(bearish): 'bearish',
            int(neutral): 'neutral'
        }
        print(df['state'].unique())
        df['market_state'] = df['state'].map(label_map)
        return df, label_map

    def save_model(self):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load_model(self, filepath):
        if os.path.exists(filepath):
            model = joblib.load(filepath)
            print("Model loaded successfully.")
            return model
        else:
            print("Model file not found.")
            return None
        
    def predict(self, predict_df, today_index):

        today_df = predict_df.iloc[today_index]

        if today_index == 0:
            yesterday_df = predict_df.iloc[today_index]
        elif today_index == len(predict_df) - 1:
            yesterday_df = predict_df.iloc[today_index - 2]
        else:
            yesterday_df = predict_df.iloc[today_index - 1]
    
        print("\nToday index: ", today_index)
        print(type(today_df))
        print(today_df)

        normalizer = ExtrernalNormalizer(stats = self.stats, full_df=self.df)
        features = normalizer.get_all_features(today_df, yesterday_df)

        new_obs = np.array([list(features.values())])

        # Predict hidden state
        state_today = self.model.predict(new_obs)
        
        market_state_today = self.market_state_labels[state_today[0]]

        return market_state_today

    def predict_probabilities(self, df, i):
        posterior_probs = self.model.predict_proba(df[self.features].values)
        
        # Get the probabilities for time step i
        if i < len(posterior_probs):
            probs = posterior_probs[i]
        else:
            probs = posterior_probs[-1]

        # Map state index to labels
        return {
            self.market_state_labels.get(idx, f'state_{idx}'): prob
            for idx, prob in enumerate(probs)
        }
    
    
