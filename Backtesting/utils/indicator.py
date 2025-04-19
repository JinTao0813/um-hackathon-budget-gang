import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator

# This class is responsible for calculating various technical indicators
class IndicatorCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def add_rsi(self, window: int = 14): # add RSI with window 14
        rsi = RSIIndicator(close=self.df["close"], window=window)
        self.df["rsi"] = rsi.rsi()
        return self

    def add_macd(self): # add MACD
        macd = MACD(close=self.df["close"])
        self.df["macd"] = macd.macd()
        return self

    def add_ema(self, window: int, label: str = None): # add EMA with custom window
        ema = EMAIndicator(close=self.df["close"], window=window)
        col_name = label if label else f"ema_{window}"
        self.df[col_name] = ema.ema_indicator()
        return self

    def add_sma(self, window: int, label: str = None): # add SMA with custom window
        sma = SMAIndicator(close=self.df["close"], window=window)
        col_name = label if label else f"sma_{window}"
        self.df[col_name] = sma.sma_indicator()
        return self

    def add_volatility(self, window: int = 20): # add volatility with window 20
        self.df["volatility"] = self.df["close"].rolling(window=window).std()
        return self

    def add_returns(self): # add log returns
        # self.df["return_pct"] = self.df["close"].pct_change() * 100
        self.df["log_return"] = (self.df["close"] / self.df["close"].shift(1)).apply(
            lambda x: pd.NA if x <= 0 else np.log(x)
        )
        return self


    def add_candle_stats(self): # add candle statistics
        self.df['ohlc_mean'] = (self.df['open'] + self.df['high'] + self.df['low'] + self.df['close']) / 4
        self.df['price_range'] = self.df['high'] - self.df['low']
        self.df['candle_body'] = abs(self.df['close'] - self.df['open'])
        self.df['direction'] = np.where(self.df['close'] > self.df['open'], 1, -1)
        return self

    def add_rolling_volatility(self, window: int = 14): # add rolling volatility with window 14
        self.df['rolling_volatility'] = self.df['log_return'].rolling(window=window).std()
        return self

    def add_volume_stats(self, span: int = 14): # add volume statistics
        self.df['volume_ema'] = self.df['volume'].ewm(span=span).mean()
        self.df['volume_spike'] = (self.df['volume'] - self.df['volume_ema']) / self.df['volume_ema']
        return self

    def add_bollinger_bands(self, window: int = 20): # add Bollinger Bands with window 20
        std = self.df["close"].rolling(window=window).std()
        sma_col = f"sma_{window}"
        if sma_col not in self.df.columns:
            self.add_sma(window=window)  # Ensure SMA is computed
        sma = self.df[sma_col]
        self.df["bollinger_upper"] = sma + 2 * std
        self.df["bollinger_lower"] = sma - 2 * std
        self.df["bollinger_width"] = (self.df["bollinger_upper"] - self.df["bollinger_lower"]) / sma
        return self

    def get_df(self): # return the dataframe with all indicators
        return self.df

