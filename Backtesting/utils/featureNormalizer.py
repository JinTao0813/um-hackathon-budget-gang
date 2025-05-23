import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Normalization classes for features in the dataset

# Self normalization means that the normalization is done using the same dataset
class SelfNormalizer: 
    def __init__(self, df):
        self.scaler = StandardScaler()
        self.df = df.copy()

    def normalize(self, features_to_normalize):
        for feature in features_to_normalize:
            norm_col = f"{feature}_norm"
            self.df[norm_col] = self.scaler.fit_transform(self.df[[feature]])
        return self.df


# External normalization means that the normalization is done using a different dataset
class ExtrernalNormalizer:
    def __init__(self, stats: pd.DataFrame, full_df: pd.DataFrame): # stats is a DataFrame with mean and std for selected columns
        """
        stats: DataFrame with mean and std for selected columns (e.g., ['mean', 'std'] as index)
        full_df: Original full dataset (for features not in stats)
        """
        self.stats = stats
        self.df = full_df

    def normalize(self, today_df: pd.Series, feature: str) -> float: # normalize a feature using precomputed stats
        """Standardize using precomputed stats (mean, std)"""
        mean = self.stats.loc[feature, 'mean']
        std = self.stats.loc[feature, 'std']
        return (today_df[feature] - mean) / std

    def normalize_from_df(self, today_df: pd.Series, feature: str) -> float: # normalize a feature using the current df
        """Standardize using current df column (fallback for non-stat features)"""
        mean = self.df[feature].mean()
        std = self.df[feature].std()
        return (today_df[feature] - mean) / std

    def compute_log_return(self, today_df: pd.Series, yesterday_df: pd.Series) -> float: # compute log return
        return np.log(today_df['close'] / yesterday_df['close'])

    def get_all_features(self, today_df: pd.Series, yesterday_df: pd.Series) -> dict:
        return { # normalized features
            'log_return': self.compute_log_return(today_df, yesterday_df),
            'volume_norm': self.normalize(today_df, 'volume'),
            'rsi_norm': self.normalize(today_df, 'rsi'),
            'macd_norm': self.normalize(today_df, 'macd'),
            'ema12_norm': self.normalize(today_df, 'ema_12'),
            'ema26_norm': self.normalize(today_df, 'ema_26'),
            'sma20_norm': self.normalize(today_df, 'sma_20'),
            'volatility_norm': self.normalize(today_df, 'volatility'),
            'ohlc_mean_norm': self.normalize_from_df(today_df, 'ohlc_mean'),
            'price_range_norm': self.normalize_from_df(today_df, 'price_range'),
            'candle_body_norm': self.normalize_from_df(today_df, 'candle_body'),
            'direction_norm': self.normalize_from_df(today_df, 'direction'),
            'rolling_volatility_norm': self.normalize_from_df(today_df, 'rolling_volatility'),
            'volume_ema_norm': self.normalize_from_df(today_df, 'volume_ema'),
            'volume_spike_norm': self.normalize_from_df(today_df, 'volume_spike'),
            'bollinger_upper_norm': self.normalize_from_df(today_df, 'bollinger_upper'),
            'bollinger_lower_norm': self.normalize_from_df(today_df, 'bollinger_lower'),
            'bollinger_width_norm': self.normalize_from_df(today_df, 'bollinger_width'),
        }
        
    def get_sentiment_features(self, sentiment_data: dict) -> dict: # get sentiment features
        """
        Get normalized sentiment features from sentiment data.
        
        Parameters:
        -----------
        sentiment_data : dict
            Dictionary containing sentiment data for a specific date
            
        Returns:
        --------
        dict
            Dictionary with normalized sentiment features
        """
        if not sentiment_data:
            return {}
            
        # Initialize with original values
        features = {
            'sentiment_mean': sentiment_data.get('sentiment_mean', 0),
            'sentiment_momentum': sentiment_data.get('sentiment_momentum', 0),
            'positive_ratio': sentiment_data.get('positive_ratio', 0.5),
            'negative_ratio': sentiment_data.get('negative_ratio', 0.5),
        }
        
        # Calculate additional derived features
        features['sentiment_signal_strength'] = abs(features['sentiment_mean']) * (1 + abs(features['sentiment_momentum']))
        features['sentiment_confidence'] = 1 - min(features['positive_ratio'], features['negative_ratio'])
        
        return features
