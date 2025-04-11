from .base import DataFetcher
import requests
import os
import time
from datetime import datetime
import pandas as pd


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
