import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
import requests
import time
import os
from datetime import datetime
import joblib
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator, MACD


## Data Fetching

class DataFetcher(ABC):
    def __init__(self, api_key, base_url, limit=1000):
        self.api_key = api_key
        self.base_url = base_url
        self.limit = limit
        self.headers = {"X-API-KEY": self.api_key}
        self.data = []

    @abstractmethod
    def fetch(self, *args, **kwargs):
        pass

    def to_dataframe(self):
        df = pd.DataFrame(self.data)
        if 'timestamp' in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
            df.set_index("timestamp", inplace=True)
        return df

    @abstractmethod
    def save_to_csv(self, *args, **kwargs):
        pass

class MarketDataFetcher(DataFetcher):
    def __init__(self, api_key, base_url, symbol, limit=1000):
        super().__init__(api_key, base_url, limit)
        self.symbol = symbol
        self.saved_filepath = None

    def fetch(self, start_time, end_time, interval, sleep_time = 0.5):
        print(f"Fetching {interval} data for {self.symbol} from {datetime.utcfromtimestamp(start_time/1000)} to {datetime.utcfromtimestamp(end_time/1000)}...")

        while start_time < end_time:
            params = {
                "symbol": self.symbol, 
                "interval": interval, 
                "start_time": start_time, 
                "limit": self.limit}

            response = requests.get(self.base_url, headers=self.headers, params=params)

            if response.status_code == 200:
                response_json = response.json()
                data = response_json.get("data", [])
                
                # Check data structure carefully here:
                if not data or not isinstance(data, list):
                    print("No more data returned or invalid format.")
                    break

                for candle in data:
                    formatted_candle = {
                        "timestamp": candle["start_time"],
                        "open": candle["open"],
                        "high": candle["high"],
                        "low": candle["low"],
                        "close": candle["close"],
                        "volume": candle["volume"]
                    }
                    self.data.append(formatted_candle)

                print(f"✅ Retrieved {len(data)} candles. Total: {len(data)}")
                start_time = data[-1]["start_time"] + 60 * 60 * 1000  # 1 hour increment
                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        self.save_to_csv(start_time, end_time, interval)
            

    def save_to_csv(self, start_time, end_time, interval):
        df = self.to_dataframe()
        os.makedirs("../datasets", exist_ok=True)
        csv_path = f"../datasets/{self.symbol}_{interval}_Training data_{start_time}_to_{end_time}.csv"
        self.saved_filepath = csv_path
        df.to_csv(csv_path)
        print(f"💾 Saved to {csv_path} with {len(df)} rows.")
        print(df.tail())

class OnChainMetricsFetcher(DataFetcher):
    def __init__(self, api_key, base_url, currency, metric, exchange, limit=1000):
        super().__init__(api_key, base_url, limit)
        self.currency = currency.lower()
        self.metric = metric.lower()
        self.exchange = exchange


    def fetch(self, window, start_time, end_time, sleep_time=0.5):
        print(f"Fetching {self.metric} data for {self.currency} from {self.exchange} for the window {window} "
              f"from {datetime.utcfromtimestamp(start_time/1000)} to {datetime.utcfromtimestamp(end_time/1000)}...")

        while start_time < end_time:
            # Construct the endpoint dynamically based on currency and metric
            endpoint = f"{self.currency}/{self.metric}"

            params = {
                "exchange": self.exchange,
                "window": window,
                "start_time": start_time,
                "end_time": end_time,
                "limit": self.limit
            }

            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)

            if response.status_code == 200:
                response_json = response.json()
                data = response_json.get("data", [])
                
                if not data or not isinstance(data, list):
                    print("No more data returned or invalid format.")
                    break

                # Process and format the fetched data
                for entry in data:
                    formatted_entry = {
                        "timestamp": entry["timestamp"],
                        "metricname": entry["value"]  # Adjust this to match the response data structure
                    }
                    self.data.append(formatted_entry)

                print(f"✅ Retrieved {len(data)} records. Total: {len(self.data)}")

                # Update the start_time for pagination or next request
                start_time = data[-1]["timestamp"] + 1000  # Increment by 1 second for next batch
                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break
        
        # Convert the collected data into a DataFrame
        return self.to_dataframe()

    def save_to_csv(self, df, exchange, window, start_time, end_time):
        os.makedirs(f"datasets/{self.currency}/{self.metric}", exist_ok=True)
        csv_filename = f"{self.currency}_{self.metric}_{exchange}_{window}_from_{start_time}_to_{end_time}.csv"
        csv_path = f"datasets/{self.currency}/{self.metric}/{csv_filename}"
        df.to_csv(csv_path)
        print(f"💾 Saved to {csv_path} with {len(df)} rows.")
        print(df.tail())


## Models
class HMM():
    def __init__(self, training_data_filepath):
        self.model_path = '../models/hmm.pkl'
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
        joblib.dump(self.model, '../models/hmm.pkl')

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


## Strategy

class Strategy:
    def __init__(self, training_dataset_filepath, predict_dataset_filepath):
        self.training_data = training_dataset_filepath
        self.predict_data = predict_dataset_filepath
        self.signals = []
        
    def generate_signals(self):
        raise NotImplementedError("This method should be overridden by subclasses")

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


class NLPStrategy(Strategy):
    print("NLP strategy initialized")
    
class EnsembleStrategy(Strategy):
    def __init__(self, data, strategies, vote_rule="majority"):
        super().__init__(data)
        self.strategies = strategies  # List of Strategy instances
        self.vote_rule = vote_rule

    def generate_signals(self):
        # Generate signals from each sub-strategy
        for strategy in self.strategies:
            strategy.generate_signals()

        # Combine signals
        combined = []
        for i in range(len(self.data)):
            signals_at_i = [strat.signals[i] for strat in self.strategies if i < len(strat.signals)]

            # Apply voting rule
            vote = self.apply_vote(signals_at_i)
            combined.append(vote)

        self.signals = combined

    def apply_vote(self, signals):
        # Remove Nones
        signals = [s for s in signals if s is not None]
        if not signals:
            return None

        if self.vote_rule == "majority":
            counts = {"buy": 0, "sell": 0}
            for s in signals:
                if s in counts:
                    counts[s] += 1
            if counts["buy"] > counts["sell"]:
                return "buy"
            elif counts["sell"] > counts["buy"]:
                return "sell"
            else:
                return None  # tie

        elif self.vote_rule == "consensus":
            return signals[0] if all(s == signals[0] for s in signals) else None

        elif self.vote_rule == "weighted":
            # Add weights if needed (e.g., HMM: 0.6, NLP: 0.4)
            # Not implemented yet
            return None

        return None


## Backtesting

class Backtesting:
    def __init__(self, data_filepath, strategy, initial_cash, max_holding_period, trading_fees):
        self.data = pd.read_csv(data_filepath)
        self.strategy = strategy
        self.cash = initial_cash
        self.position = 0
        self.entry_price = 0
        self.entry_index = 0
        self.signals = []
        self.trades = []
        self.equity = 0
        self.equity_curve = []
        self.holding_period = max_holding_period
        self.trading_fees = trading_fees
        self.portfolio_values = []
        self.trade_logs = []

    def run(self):
        self.strategy.generate_signals() 
        self.signals = self.strategy.signals
        for i in range(len(self.signals)):
            current_data = self.data.iloc[i]

            date = current_data.get('date') or current_data.get('datetime') or i
            close = current_data['close']
            premium_index = current_data.get('premium_index')
            trade = None
            pnl = 0

            signal = self.signals[i] if i<len(self.signals) else None
            price = current_data['close']
            # BUY action
            if signal == 'buy' and self.cash > price and self.position == 0: # Buy if buy signal occurs && we have cash && we don't have a position
                self.position = self.cash // (price * (1 + self.trading_fees)) # Calculate the number of shares we can buy
                cost = self.position * price * (1 + self.trading_fees)
                self.cash -= cost
                self.entry_price = price
                self.entry_index = i
                trade = 'buy'
                self.trades.append((trade, price, i))

            # SELL action
            elif self.signals[i] == 'sell' and self.position > 0: # Sell if sell signal occurs && we have a position
                if self.entry_index is not None and i - self.entry_index > self.holding_period:
                    sell_order = self.position * price * (1 - self.trading_fees)
                    self.cash += sell_order
                    trade = 'sell'
                    self.trades.append((trade, price, i))
                    self.position = 0
                    self.entry_price = 0
                    self.entry_index = None
       
            equity = self.cash + self.position * price
            self.equity_curve.append(equity)
            self.portfolio_values.append(equity)
            self.equity

             # Z-score and price change (can be strategy-specific; placeholder here)
            if i > 0:
                price_change = close - self.data['close'][i - 1]
            else:
                price_change = 0
            z_score = (price_change - np.mean(self.data['close'].diff().dropna())) / np.std(self.data['close'].diff().dropna())

            # Drawdown
            peak = max(self.equity_curve)
            drawdown = equity - peak

            # Append all fields to rows
            self.trade_logs.append({
                'datetime': date,
                'close': close,
                'premium_index': premium_index,
                'price_change': price_change,
                'z_score': z_score,
                'position': self.position,
                'trade': trade,
                'pnl': pnl,
                'equity': equity,
                'drawdown': drawdown
            })

    def get_performance_results(self):
        final_value = self.cash + self.position * self.data['close'].iloc[-1]
        total_return = (final_value - self.cash) / self.cash
        total_trades = len(self.trades)
        wins = 0
        losses = 0
        for i in range(0, len(self.trades)-1, 2):
            buy_price = self.trades[i][1]
            sell_price = self.trades[i+1][1]
            if sell_price > buy_price:
                wins += 1
            else:
                losses += 1
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0


        pv = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(pv)
        drawdown = (peak-pv) / peak
        max_drawdown = np.max(drawdown) * 100

        returns = np.diff(pv) / pv[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0

        results = {
            'Start Trade Date': self.data['timestamp'].iloc[0],
            'End Trade Date': self.data['timestamp'].iloc[-1],
            'Final Portfolio Value': final_value,
            'Total Return (%)': total_return,
            'Number of Trades': total_trades,  
            'Win Rate (%)': round(win_rate, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
        }
                
        return results
    
    def get_trade_logs_csv(self, filename = 'backtest_trade_logs.csv'):
        df = pd.DataFrame(self.trade_logs)
        df.to_csv(filename, index=False)
        print(f"Trade logs saved to {filename}")
