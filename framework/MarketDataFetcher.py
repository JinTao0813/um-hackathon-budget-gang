import requests
import pandas as pd
import time
import os
from datetime import datetime

class MarketDataFetcher:
    def __init__(self, api_key, base_url, symbol, interval, limit=1000):
        self.api_key = api_key
        self.base_url = base_url
        self.symbol = symbol
        self.interval = interval
        self.limit = limit
        self.headers = {"X-API-KEY": self.api_key}
        self.data = []

    def fetch(self, start_time, end_time, sleep_time = 0.5):
        print(f"Fetching {self.interval} data for {self.symbol} from {datetime.utcfromtimestamp(start_time/1000)} to {datetime.utcfromtimestamp(end_time/1000)}...")

        while start_time < end_time:
            params = {
                "symbol": self.symbol, 
                "interval": self.interval, 
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
                time.sleep(0.5)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break
        
        return self.to_dataframe()
            
    def to_dataframe(self):
        # === CONVERT TO DATAFRAME ===
        df = pd.DataFrame(self.data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def save_to_csv(self, df, start_time, end_time):
        # === SAVE TO CSV ===
        os.makedirs("../datasets", exist_ok=True)
        csv_path = f"../datasets/{self.symbol}_{self.interval}_Training data_{start_time}_to_{end_time}.csv"
        df.to_csv(csv_path)
        print(f"💾 Saved to {csv_path} with {len(df)} rows.")

        # === OPTIONAL: Preview ===
        print(df.tail())
