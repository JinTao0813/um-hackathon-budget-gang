from .base import DataFetcher
import requests
import os
import time
from datetime import datetime
import pandas as pd

class CoinbaseFetcher(DataFetcher):
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
        os.makedirs("datasets", exist_ok=True)
        csv_path = f"datasets/{self.symbol}_{interval}_Training data_{start_time}_to_{end_time}.csv"
        self.saved_filepath = csv_path
        df.to_csv(csv_path)
        print(f"💾 Saved to {csv_path} with {len(df)} rows.")
        print(df.tail())