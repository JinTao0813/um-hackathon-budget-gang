import Backtest
import DataFetcher
import pandas as pd
import numpy as np
from datetime import datetime

# Example usage
if __name__ == "__main__":

    # Fetching data from the API
    API_KEY = "GlYxSZP9hnooNl6gGAjtkptkeqehSnk5C60Akhpw5zupBK6O"
    BASE_URL = "https://api.datasource.cybotrade.rs/coinbase/candle"
    SYMBOL = "BTC-USD"
    INTERVAL = "1h"
    LIMIT = 1000

    start_time = int(datetime(2018, 1, 1).timestamp() * 1000)
    end_time = int(datetime(2020, 12, 31, 23, 59).timestamp() * 1000)
    
    fetcher = DataFetcher(API_KEY, BASE_URL, SYMBOL, INTERVAL, LIMIT)
    df = fetcher.fetch(start_time, end_time)
    df.save_to_csv(df, start_time, end_time)

    data = pd.read_csv('../datasets/BTCUSDT_backtest_features.csv')

    # Training the model
    hmm = Backtest.HMMStrategy (data)
    nlp = Backtest.NLPStrategy (data)

    # Combining strategies
    ensemble = Backtest.EnsembleStrategy (data, [hmm, nlp], vote_rule="majority")

    # Running the backtest
    bt = Backtest.Backtesting(data, ensemble, initial_capital=10000, trading_fees=0.006, max_holding_period=10)
    bt.run()
    print(bt.get_performance_results)