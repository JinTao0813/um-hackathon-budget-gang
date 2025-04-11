import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import os
import joblib
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD



class HMM():
    def __init__(self, training_data_filepath):
        self.model_path = 'models/hmm.pkl'
        self.model = None
        self.training_data = pd.read_csv(training_data_filepath)
        self.df = self.preprocess_data(self.training_data)
        self.market_state_labels = {}
        self.stats = None
        self.initialize_metrics()

    def preprocess_data(self, df):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        df = df.dropna()

        # Calculate additional metrics
        
        rsi = RSIIndicator(close=df["close"], window=14)
        df["rsi"] = rsi.rsi()

        macd = MACD(close=df["close"])
        df["macd"] = macd.macd()

        ema12 = EMAIndicator(close=df["close"], window=12)
        ema26 = EMAIndicator(close=df["close"], window=26)
        df["ema_12"] = ema12.ema_indicator()
        df["ema_26"] = ema26.ema_indicator()

        sma20 = SMAIndicator(close=df["close"], window=20)
        df["sma_20"] = sma20.sma_indicator()

        df["volatility"] = df["close"].rolling(window=20).std()

        df["return_pct"] = df["close"].pct_change() * 100
        df["log_return"] = (df["close"] / df["close"].shift(1)).apply(lambda x: pd.NA if x <= 0 else np.log(x))


        df['ohlc_mean'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['price_range'] = df['high'] - df['low']
        df['candle_body'] = abs(df['close'] - df['open'])
        df['direction'] = np.where(df['close'] > df['open'], 1, -1)
        df['rolling_volatility'] = df['log_return'].rolling(window=14).std()
        df['volume_ema'] = df['volume'].ewm(span=14).mean()
        df['volume_spike'] = (df['volume'] - df['volume_ema']) / df['volume_ema']
        df['bollinger_upper'] = df['sma_20'] + 2 * df['close'].rolling(window=20).std()
        df['bollinger_lower'] = df['sma_20'] - 2 * df['close'].rolling(window=20).std()
        df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['sma_20']
        
        df = df.dropna()  # Drop again after feature creation
        return df

    def initialize_metrics(self):
        metrics = ['volume', 'rsi', 'macd', 'ema_12', 'ema_26', 'sma_20', 'volatility']
        self.stats = self.df[metrics].agg(['mean', 'std']).T


    def train(self):
        features_to_normalize = [
            'volume', 'rsi', 'macd', 'ema_12', 'ema_26', 'sma_20', 'volatility',
            'ohlc_mean', 'price_range', 'candle_body', 'direction',
            'rolling_volatility', 'volume_ema', 'volume_spike',
            'bollinger_upper', 'bollinger_lower', 'bollinger_width'
        ]
        df = self.normalize_features(self.df, features_to_normalize)
        

        observation_columns = ['log_return'] + [f"{feat}_norm" for feat in features_to_normalize]
        observations = self.extract_observations(df, observation_columns)
        
        self.model = self.train_hmm_model(observations)
        df = self.assign_states(df, self.model, observation_columns)
        
        df, state_labels = self.identify_market_states(df)
        
        self.df = df
        self.market_state_labels = state_labels
        
        print("Converged:", self.model.monitor_.converged)
        print("Final log likelihood:", self.model.monitor_.history[-1])
        print("State Labels:", self.market_state_labels)
        print(df[['timestamp', 'open', 'close', 'log_return', 'state', 'market_state']])
        
        self.save_model()


    def normalize_features(self, df, features_to_normalize):
        scaler = StandardScaler()
        for feature in features_to_normalize:
            norm_col = f"{feature}_norm"
            df[norm_col] = scaler.fit_transform(df[[feature]])
        return df


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
        state_analysis = df.groupby('state').agg({'log_return': 'mean'})
        bullish = state_analysis['log_return'].idxmax()
        bearish = state_analysis['log_return'].idxmin()
        neutral = list(set(df['state'].unique()) - {bullish, bearish})[0]
        
        label_map = {
            int(bullish): 'bullish',
            int(bearish): 'bearish',
            int(neutral): 'neutral'
        }
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

        # Calculate features
        log_return = np.log(today_df['close']/ yesterday_df['close'])
        volume_norm = (today_df['volume'] - self.stats.loc['volume', 'mean']) / self.stats.loc['volume', 'std']
        rsi_norm = (today_df['rsi'] - self.stats.loc['rsi', 'mean']) / self.stats.loc['rsi', 'std']
        macd_norm = (today_df['macd'] - self.stats.loc['macd', 'mean']) / self.stats.loc['macd', 'std']
        ema12_norm = (today_df['ema_12'] - self.stats.loc['ema_12', 'mean']) / self.stats.loc['ema_12', 'std']
        ema26_norm = (today_df['ema_26']  - self.stats.loc['ema_26', 'mean']) / self.stats.loc['ema_26', 'std']
        sma20_norm = (today_df['sma_20']  - self.stats.loc['sma_20', 'mean']) / self.stats.loc['sma_20', 'std']
        volatility_norm = (today_df['volatility']  - self.stats.loc['volatility', 'mean']) / self.stats.loc['volatility', 'std']
        ohlc_mean_norm = (today_df['ohlc_mean']  - self.df['ohlc_mean'].mean()) / self.df['ohlc_mean'].std()
        price_range_norm = (today_df['price_range'] - self.df['price_range'].mean()) / self.df['price_range'].std()
        candle_body_norm = (today_df['candle_body']  - self.df['candle_body'].mean()) / self.df['candle_body'].std()
        direction_norm = (today_df['direction']  - self.df['direction'].mean()) / self.df['direction'].std()    
        rolling_volatility_norm = (today_df['rolling_volatility']  - self.df['rolling_volatility'].mean()) / self.df['rolling_volatility'].std()
        volume_ema_norm = (today_df['volume_ema']  - self.df['volume_ema'].mean()) / self.df['volume_ema'].std()
        volume_spike_norm = (today_df['volume_spike']  - self.df['volume_spike'].mean()) / self.df['volume_spike'].std()
        bollinger_upper_norm = (today_df['bollinger_upper']  - self.df['bollinger_upper'].mean()) / self.df['bollinger_upper'].std()
        bollinger_lower_norm = (today_df['bollinger_lower']  - self.df['bollinger_lower'].mean()) / self.df['bollinger_lower'].std()
        bollinger_width_norm = (today_df['bollinger_width']  - self.df['bollinger_width'].mean()) / self.df['bollinger_width'].std()

        # Format as 2D array
        new_obs = np.array([[log_return, volume_norm, rsi_norm, macd_norm, ema12_norm, ema26_norm, sma20_norm, volatility_norm,
                             ohlc_mean_norm, price_range_norm, candle_body_norm, direction_norm,
                             rolling_volatility_norm, volume_ema_norm, volume_spike_norm,
                             bollinger_upper_norm, bollinger_lower_norm, bollinger_width_norm]])

        # Predict hidden state
        state_today = self.model.predict(new_obs)
        
        market_state_today = self.market_state_labels[state_today[0]]

        return market_state_today
