from .base import DataFetcher
import requests
import os
import time
from datetime import datetime
import pandas as pd

# Fetcher for CryptoQuant data
class CryptoQuantFetcher(DataFetcher):
    def __init__(self, api_key, base_url, currency, endpoint_category, metric, exchange, limit= any):
        super().__init__(api_key, base_url, limit)
        self.currency = currency.lower()
        self.endpoint_category = endpoint_category.lower() # e.g., "exchange" or "onchain"
        self.metric = metric.lower()
        self.exchange = exchange # e.g., "binance"

    def fetch(self):
        print("fetching data...")

    @staticmethod
    def merge_selected_csv_files(csv_files, output_filename, start_ts=None, end_ts=None):
        """
        Merges selected CSV files into one, aligning on a fixed daily timestamp index.
        Useful for consistent merging of mixed-frequency time series.
        Converts all non-timestamp columns to float for normalization.
        """

        all_data = []

        for full_path in csv_files:
            if not os.path.exists(full_path):
                print(f"❌ File not found: {full_path}")
                continue

            df = pd.read_csv(full_path)

            if "timestamp" not in df.columns:
                print(f"⚠️ Skipping {full_path} - No 'timestamp' column found") 
                continue

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

            # Convert all non-timestamp columns to float
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Resample to daily if not already daily
            freq = df.index.to_series().diff().mode()[0]
            is_hourly = freq <= pd.Timedelta(hours=1.5)

            if is_hourly: # Resample to daily
                df = df.resample("1D").ffill()

            all_data.append(df)

            # Automatically infer min/max timestamp if not specified
            if start_ts is None or pd.to_datetime(start_ts) > df.index.min():
                start_ts = df.index.min() # Set to the earliest timestamp in the data
            if end_ts is None or pd.to_datetime(end_ts) < df.index.max():
                end_ts = df.index.max()

        if not all_data:
            print("❌ No valid CSVs merged.")
            return

        # Create fixed daily index
        time_index = pd.date_range(start=start_ts, end=end_ts, freq="1D") # Daily frequency
        merged_df = pd.DataFrame(index=time_index)

        for df in all_data:
            merged_df = merged_df.join(df, how="outer")

        # Reset index to have timestamp as a column again
        merged_df.reset_index(inplace=True)
        merged_df.rename(columns={"index": "timestamp"}, inplace=True)

        # ❌ Drop rows with any missing data
        merged_df.dropna(inplace=True)
        merged_df.drop_duplicates(subset=["timestamp"], inplace=True)

        # Save result
        os.makedirs("datasets", exist_ok=True)
        output_path = output_filename
        merged_df.to_csv(output_path, index=False, float_format="%.10f")
        print(f"✅ Merged daily CSV saved to: {output_path}")
        return output_path


    def fetch_ohlcv(self, window, start_time=None, end_time=None, sleep_time=0.5): # Fetch OHLCV data from CryptoQuant
        print(f"Fetching {self.endpoint_category}/{self.metric} data for {self.currency} from {self.exchange} with window '{window}'...")

        endpoint = f"{self.currency}/{self.endpoint_category}/{self.metric}" # e.g., "exchange/ohlcv"
        fetching_by_range = start_time is not None and end_time is not None # Check if both start_time and end_time are provided
        remaining_limit = self.limit if not fetching_by_range else float('inf')  # Don't apply hard limit when using a range

        while remaining_limit > 0:
            params = {
                "market": "spot", # e.g., "spot" or "futures"
                "symbol": "btc_usdt", # e.g., "btc_usdt"
                "window": window, # e.g., "1h"
                "exchange": self.exchange # e.g., "binance"
            }

            if fetching_by_range: # If both start_time and end_time are provided
                params["start_time"] = start_time
                params["end_time"] = end_time
                print(f"Using range: {datetime.utcfromtimestamp(start_time / 1000)} to {datetime.utcfromtimestamp(end_time / 1000)}")
            elif start_time: # If only start_time is provided
                params["start_time"] = start_time
                print(f"Using start_time + limit={params['limit']} from {datetime.utcfromtimestamp(start_time / 1000)}")
            elif end_time: # If only end_time is provided
                params["end_time"] = end_time
                print(f"Using end_time + limit={params['limit']} to {datetime.utcfromtimestamp(end_time / 1000)}")
            else:
                raise ValueError("You must provide at least 'start_time' or 'end_time'.")

            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)

            if response.status_code == 200:
                data = response.json().get("data", [])
                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry["start_time"],
                        "open": entry.get("open"),
                        "high": entry.get("high"),
                        "low": entry.get("low"),
                        "close": entry.get("close"),
                        "volume": entry.get("volume")
                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                last_timestamp = data[-1]["start_time"]

                # Stop if we've passed the end_time
                if fetching_by_range and (last_timestamp >= end_time): # If using a range and we've passed the end_time
                    break

                # Move forward
                start_time = last_timestamp + 1000  # Move 1 second forward (timestamp is in ms)
                if not fetching_by_range:
                    remaining_limit -= len(data)

                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.exchange, self.metric, window, start_time, end_time)

    # Fetch netflow data from CryptoQuant
    def fetch_netflow(self, window, start_time=None, end_time=None, sleep_time=0.5):
        print(f"Fetching {self.endpoint_category}/{self.metric} data for {self.currency} from {self.exchange} with window '{window}'...")

        endpoint = f"{self.currency}/{self.endpoint_category}/{self.metric}"
        fetching_by_range = start_time is not None and end_time is not None
        remaining_limit = self.limit if not fetching_by_range else float('inf')  # Don't apply hard limit when using a range

        while remaining_limit > 0:
            params = {
                "window": window, # e.g., "1h"
                "exchange": self.exchange # e.g., "binance"
            }

            if fetching_by_range: # If both start_time and end_time are provided
                params["start_time"] = start_time # e.g., 1690000000000
                params["end_time"] = end_time 
                print(f"Using range: {datetime.utcfromtimestamp(start_time / 1000)} to {datetime.utcfromtimestamp(end_time / 1000)}")
            elif start_time: # If only start_time is provided
                params["start_time"] = start_time 
                print(f"Using start_time + limit={params['limit']} from {datetime.utcfromtimestamp(start_time / 1000)}")
            elif end_time: # If only end_time is provided
                params["end_time"] = end_time
                print(f"Using end_time + limit={params['limit']} to {datetime.utcfromtimestamp(end_time / 1000)}")
            else:
                raise ValueError("You must provide at least 'start_time' or 'end_time'.")

            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)

            if response.status_code == 200:
                data = response.json().get("data", [])
                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry["start_time"],
                        "netflow_total": entry.get("netflow_total")  # Adjust as needed
                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                last_timestamp = data[-1]["start_time"]

                # Stop if we've passed the end_time
                if fetching_by_range and (last_timestamp >= end_time):
                    break

                # Move forward
                start_time = last_timestamp + 1000  # Move 1 second forward (timestamp is in ms)
                if not fetching_by_range:
                    remaining_limit -= len(data)

                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.exchange, self.metric, window, start_time, end_time)

    # Fetch exchange whale ratio data from CryptoQuant
    def fetch_exchange_whale_ratio(self, window, start_time=None, end_time=None, sleep_time=0.5):
        print(f"Fetching {self.endpoint_category}/{self.metric} data for {self.currency} from {self.exchange} with window '{window}'...")

        endpoint = f"{self.currency}/{self.endpoint_category}/{self.metric}"
        fetching_by_range = start_time is not None and end_time is not None
        remaining_limit = self.limit if not fetching_by_range else float('inf')  # Don't apply hard limit when using a range

        while remaining_limit > 0:
            params = {
                "window": window,
                "exchange": self.exchange
            }

            if fetching_by_range:
                params["start_time"] = start_time
                params["end_time"] = end_time
                print(f"Using range: {datetime.utcfromtimestamp(start_time / 1000)} to {datetime.utcfromtimestamp(end_time / 1000)}")
            elif start_time:
                params["start_time"] = start_time
                print(f"Using start_time + limit={params['limit']} from {datetime.utcfromtimestamp(start_time / 1000)}")
            elif end_time:
                params["end_time"] = end_time
                print(f"Using end_time + limit={params['limit']} to {datetime.utcfromtimestamp(end_time / 1000)}")
            else:
                raise ValueError("You must provide at least 'start_time' or 'end_time'.")

            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)

            if response.status_code == 200:
                data = response.json().get("data", [])
                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry["start_time"],
                        "exchange_whale_ratio": entry.get("exchange_whale_ratio")  # Adjust as needed
                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                last_timestamp = data[-1]["start_time"]

                # Stop if we've passed the end_time
                if fetching_by_range and (last_timestamp >= end_time):
                    break

                # Move forward
                start_time = last_timestamp + 1000  # Move 1 second forward (timestamp is in ms)
                if not fetching_by_range:
                    remaining_limit -= len(data)

                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.exchange, self.metric, window, start_time, end_time)
    
    # Fetch funding rates data from CryptoQuant
    def fetch_funding_rates(self, window, start_time=None, end_time=None, sleep_time=0.5):
        print(f"Fetching {self.endpoint_category}/{self.metric} data for {self.currency} from {self.exchange} with window '{window}'...")

        endpoint = f"{self.currency}/{self.endpoint_category}/{self.metric}"
        fetching_by_range = start_time is not None and end_time is not None
        remaining_limit = self.limit if not fetching_by_range else float('inf')  # Don't apply hard limit when using a range

        while remaining_limit > 0:
            params = {
                "window": window, # e.g., "1h"
                "exchange": self.exchange # e.g., "binance"
            }

            if fetching_by_range:
                params["start_time"] = start_time
                params["end_time"] = end_time
                print(f"Using range: {datetime.utcfromtimestamp(start_time / 1000)} to {datetime.utcfromtimestamp(end_time / 1000)}")
            elif start_time:
                params["start_time"] = start_time
                print(f"Using start_time + limit={params['limit']} from {datetime.utcfromtimestamp(start_time / 1000)}")
            elif end_time:
                params["end_time"] = end_time
                print(f"Using end_time + limit={params['limit']} to {datetime.utcfromtimestamp(end_time / 1000)}")
            else:
                raise ValueError("You must provide at least 'start_time' or 'end_time'.")

            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)

            if response.status_code == 200:
                data = response.json().get("data", [])
                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry["start_time"],
                        "funding_rates": entry.get("funding_rates")  # Adjust as needed
                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                last_timestamp = data[-1]["start_time"]

                # Stop if we've passed the end_time
                if fetching_by_range and (last_timestamp >= end_time):
                    break

                # Move forward
                start_time = last_timestamp + 1000  # Move 1 second forward (timestamp is in ms)
                if not fetching_by_range:
                    remaining_limit -= len(data)

                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.exchange, self.metric, window, start_time, end_time)

    def save_to_csv(self, exchange, metric, window, start_time, end_time):
        df = self.to_dataframe()
        os.makedirs("datasets", exist_ok=True)
        csv_path = f"datasets/{exchange}_{metric}_{window}_Training data_{start_time}_to_{end_time}.csv"
        self.saved_filepath = csv_path
        df.to_csv(csv_path)
        print(f"💾 Saved to {csv_path} with {len(df)} rows.")
        print(df.tail())

# Fetcher for CryptoQuant data without exchange (meaning parameters are not exchange-specific)
class CryptoQuantFetcherWithoutExchange(DataFetcher):
    def __init__(self, api_key, base_url, currency, endpoint_category, metric, limit=1000):
        super().__init__(api_key, base_url, limit)
        self.currency = currency.lower()
        self.endpoint_category = endpoint_category.lower()
        self.metric = metric.lower()

    def fetch(self):
        print("fetching data...")

    def fetch_dormancy(self, window, start_time=None, end_time=None, sleep_time=0.5):
        print(f"Fetching {self.endpoint_category}/{self.metric} data for {self.currency} with window '{window}'...")

        endpoint = f"{self.currency}/{self.endpoint_category}/{self.metric}"
        fetching_by_range = start_time is not None and end_time is not None
        remaining_limit = self.limit if not fetching_by_range else float('inf')  # Don't apply hard limit when using a range

        while remaining_limit > 0:
            params = {
                "window": window
            }

            if fetching_by_range:
                params["start_time"] = start_time
                params["end_time"] = end_time
                print(f"Using range: {datetime.utcfromtimestamp(start_time / 1000)} to {datetime.utcfromtimestamp(end_time / 1000)}")
            elif start_time:
                params["start_time"] = start_time
                print(f"Using start_time + limit={params['limit']} from {datetime.utcfromtimestamp(start_time / 1000)}")
            elif end_time:
                params["end_time"] = end_time
                print(f"Using end_time + limit={params['limit']} to {datetime.utcfromtimestamp(end_time / 1000)}")
            else:
                raise ValueError("You must provide at least 'start_time' or 'end_time'.")

            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)

            if response.status_code == 200:
                data = response.json().get("data", [])
                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry["start_time"],
                        "average_dormancy": entry.get("average_dormancy"),
                        "sa_average_dormancy": entry.get("sa_average_dormancy")
                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                last_timestamp = data[-1]["start_time"]

                # Stop if we've passed the end_time
                if fetching_by_range and (last_timestamp >= end_time):
                    break

                # Move forward
                start_time = last_timestamp + 1000  # Move 1 second forward (timestamp is in ms)
                if not fetching_by_range:
                    remaining_limit -= len(data)

                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.metric, window, start_time, end_time)

    def save_to_csv(self, metric, window, start_time, end_time):
        df = self.to_dataframe()
        os.makedirs("datasets", exist_ok=True)
        csv_path = f"datasets/{metric}_{window}_Training data_{start_time}_to_{end_time}.csv"
        self.saved_filepath = csv_path
        df.to_csv(csv_path)
        print(f"💾 Saved to {csv_path} with {len(df)} rows.")
        print(df.tail())

# Fetcher for Glassnode data
class GlassnodeFetcher(DataFetcher):
    def __init__(
        self,
        api_key,
        base_url,
        asset,
        endpoint,
        network=None,
        since=None,
        until=None,
        interval='24h',
        format='json',
        timestamp_format='humanized',
        limit=1000
    ):
        super().__init__(api_key, base_url, limit)
        self.asset = asset
        self.endpoint = endpoint.lower()
        self.network = network
        self.since = since
        self.until = until
        self.interval = interval
        self.format = format
        self.timestamp_format = timestamp_format

    def fetch(self):
        print("fetching data...")

    def fetch_new_address(self, interval, start_time=None, end_time=None, sleep_time=0.5):
        print(f"Fetching {self.endpoint} data for {self.asset} at interval '{interval}'...")

        remaining_limit = self.limit 
        while remaining_limit > 0: # Check if we have remaining limit
            params = {
                "a": self.asset, # e.g., "BTC"
                "i": interval, # e.g., "24h"
                "start_timestamp": start_time,
                "end_timestamp": end_time,
                "f": self.format
            }

            current_batch_limit = min(remaining_limit, 1000)
            params["limit"] = current_batch_limit  # Glassnode may ignore this, but we can include it just in case

            url = f"{self.base_url}/{self.endpoint}"
            print(f"Fetching from {url} with params: {params}")
            headers = {"X-API-Key": f"{self.api_key}"}
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200: # Check if the response is successful
                try:
                    data = response.json()
                    if isinstance(data, dict) and "data" in data:
                        data = data["data"]
                except ValueError:
                    print("❌ Failed to decode JSON.")
                    break

                if not isinstance(data, list):
                    print(f"Unexpected response format: {data}")
                    break

                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry.get("t") or entry.get("start_time"),  # support both formats
                        "new_address": float(entry["v"]) if isinstance(entry["v"], str) else entry["v"]
                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                remaining_limit -= len(data)
                last_timestamp = entry.get("t") or entry.get("start_time")

                # Advance to next batch
                start_time = last_timestamp + 1
                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.endpoint, interval, start_time, end_time)

    def fetch_active_address(self, interval, start_time=None, end_time=None, sleep_time=0.5): # Fetch active address data
        print(f"Fetching {self.endpoint} data for {self.asset} at interval '{interval}'...")

        remaining_limit = self.limit
        while remaining_limit > 0:
            params = {
                "a": self.asset,
                "i": interval,
                "start_timestamp": start_time,
                "end_timestamp": end_time,
                "f": self.format
            }

            current_batch_limit = min(remaining_limit, 1000)
            params["limit"] = current_batch_limit  # Glassnode may ignore this, but we can include it just in case

            url = f"{self.base_url}/{self.endpoint}"
            print(f"Fetching from {url} with params: {params}")
            headers = {"X-API-Key": f"{self.api_key}"}
            response = requests.get(url, headers=headers, params=params)

            if response.status_code == 200:
                try:
                    data = response.json()
                    if isinstance(data, dict) and "data" in data:
                        data = data["data"]
                except ValueError:
                    print("❌ Failed to decode JSON.")
                    break

                if not isinstance(data, list): # Check if the response is a list
                    print(f"Unexpected response format: {data}")
                    break

                if not data:
                    print("No more data returned.")
                    break

                for entry in data: # Iterate through the entries
                    self.data.append({ # Append each entry to the data list
                        "timestamp": entry.get("t") or entry.get("start_time"),  # support both formats
                        "active_address": float(entry["v"]) if isinstance(entry["v"], str) else entry["v"]
                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                remaining_limit -= len(data) # Update remaining limit
                last_timestamp = entry.get("t") or entry.get("start_time")

                # Advance to next batch
                start_time = last_timestamp + 1
                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.endpoint, interval, start_time, end_time)

    def save_to_csv(self, endpoint, interval, start_time, end_time):
        df = self.to_dataframe()
        os.makedirs("datasets", exist_ok=True)

        # Replace slashes to avoid unintended subdirectories
        safe_endpoint = endpoint.replace("/", "_")

        csv_path = f"datasets/glassnode_{safe_endpoint}_{interval}_Training data_{start_time}_to_{end_time}.csv"
        self.saved_filepath = csv_path
        df.to_csv(csv_path)
        print(f"💾 Saved to {csv_path} with {len(df)} rows.")
        print(df.tail())


