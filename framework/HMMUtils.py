import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import joblib

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    df = df.dropna()

    # Calculate additional metrics
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


def normalize_features(df, features_to_normalize):
    scaler = StandardScaler()
    for feature in features_to_normalize:
        norm_col = f"{feature}_norm"
        df[norm_col] = scaler.fit_transform(df[[feature]])
    return df


def extract_observations(df, observation_columns):
    return df[observation_columns].values


def train_hmm_model(observations, n_states=3, n_iter=1000):
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=n_iter, verbose=True)
    model.fit(observations)
    return model


def assign_states(df, model, observation_columns):
    observations = extract_observations(df, observation_columns)
    df['state'] = model.predict(observations)
    return df


def label_market_states(df):
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


def save_model(model, filepath):
    joblib.dump(model, filepath)
