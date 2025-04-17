from .base import DataFetcher
import requests
import os
import time
from datetime import datetime
import pandas as pd


class CryptoQuantFetcher(DataFetcher):

    @staticmethod
    def merge_selected_csv_files(csv_files, output_filename, start_ts=None, end_ts=None):
        """
        Merges selected CSV files into one, aligning on a fixed hourly timestamp index.
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

            # Detect frequency
            freq = df.index.to_series().diff().mode()[0]
            is_daily = freq >= pd.Timedelta(hours=23)

            # Resample if daily
            if is_daily:
                df = df.resample("1H").ffill()

            all_data.append(df)

            # Automatically infer min/max timestamp if not specified
            if start_ts is None or pd.to_datetime(start_ts) > df.index.min():
                start_ts = df.index.min()
            if end_ts is None or pd.to_datetime(end_ts) < df.index.max():
                end_ts = df.index.max()

        if not all_data:
            print("❌ No valid CSVs merged.")
            return

        # Create fixed hourly index
        time_index = pd.date_range(start=start_ts, end=end_ts, freq="1H")
        merged_df = pd.DataFrame(index=time_index)

        for df in all_data:
            merged_df = merged_df.join(df, how="outer")

        # Reset index to have timestamp as a column again
        merged_df.reset_index(inplace=True)
        merged_df.rename(columns={"index": "timestamp"}, inplace=True)

        # Save result
        os.makedirs("datasets", exist_ok=True)
        output_path = os.path.join("datasets", output_filename)
        merged_df.to_csv(output_path, index=False, float_format="%.10f")
        print(f"✅ Merged CSV saved to: {output_path}")

    def __init__(self, api_key, base_url, currency, endpoint_category, metric, exchange, limit= any):
        super().__init__(api_key, base_url, limit)
        self.currency = currency.lower()
        self.endpoint_category = endpoint_category.lower()
        self.metric = metric.lower()
        self.exchange = exchange

    def fetch(self):
        print("fetching data...")

    def fetch_netflow(self, window, start_time=None, end_time=None, sleep_time=0.5):
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
    
    def fetch_funding_rates(self, window, start_time=None, end_time=None, sleep_time=0.5):
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

class CryptoQuantFetcherWithoutExchange(DataFetcher):
    def __init__(self, api_key, base_url, currency, endpoint_category, metric, limit=1000):
        super().__init__(api_key, base_url, limit)
        self.currency = currency.lower()
        self.endpoint_category = endpoint_category.lower()
        self.metric = metric.lower()

    def fetch(self):
        print("fetching data...")

    def fetch_mvrv(self, window, start_time=None, end_time=None, sleep_time=0.5):
        print(f"Fetching {self.endpoint_category}/{self.metric} data for {self.currency} with window '{window}'...")

        endpoint = f"{self.currency}/{self.endpoint_category}/{self.metric}"
        remaining_limit = self.limit  # Track how many data points you want in total

        while remaining_limit > 0:
            # Construct params based on allowed combinations
            params = {
                "window": window
            }

            # Always fetch at most 1000 (or remaining)
            current_batch_limit = min(remaining_limit, 1000)
            params["limit"] = current_batch_limit

            if start_time:
                params["start_time"] = start_time
                print(f"Using start_time + limit={current_batch_limit} from {datetime.utcfromtimestamp(start_time/1000)}")
            elif end_time:
                params["end_time"] = end_time
                print(f"Using end_time + limit={current_batch_limit} to {datetime.utcfromtimestamp(end_time/1000)}")
            else:
                raise ValueError("You must provide at least 'start_time' or 'end_time'.")

            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)

            if response.status_code == 200:
                response_json = response.json()
                data = response_json.get("data", [])

                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry["start_time"],
                        "mvrv": entry.get("mvrv")  # Adjust as needed
                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                remaining_limit -= len(data)

                # Prepare for next loop
                last_timestamp = data[-1]["start_time"]
                if end_time and (last_timestamp + 1000 > end_time):
                    break

                start_time = last_timestamp + 1000
                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.metric, window, start_time, end_time)
    
    def fetch_sopr(self, window, start_time=None, end_time=None, sleep_time=0.5):
        print(f"Fetching {self.endpoint_category}/{self.metric} data for {self.currency} with window '{window}'...")

        endpoint = f"{self.currency}/{self.endpoint_category}/{self.metric}"
        remaining_limit = self.limit  # Track how many data points you want in total

        while remaining_limit > 0:
            # Construct params based on allowed combinations
            params = {
                "window": window
            }

            # Always fetch at most 1000 (or remaining)
            current_batch_limit = min(remaining_limit, 1000)
            params["limit"] = current_batch_limit

            if start_time:
                params["start_time"] = start_time
                print(f"Using start_time + limit={current_batch_limit} from {datetime.utcfromtimestamp(start_time/1000)}")
            elif end_time:
                params["end_time"] = end_time
                print(f"Using end_time + limit={current_batch_limit} to {datetime.utcfromtimestamp(end_time/1000)}")
            else:
                raise ValueError("You must provide at least 'start_time' or 'end_time'.")

            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)

            if response.status_code == 200:
                response_json = response.json()
                data = response_json.get("data", [])

                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry["start_time"],
                        "sopr_ratio": entry.get("sopr_ratio")  # Adjust as needed
                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                remaining_limit -= len(data)

                # Prepare for next loop
                last_timestamp = data[-1]["start_time"]
                if end_time and (last_timestamp + 1000 > end_time):
                    break

                start_time = last_timestamp + 1000
                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.metric, window, start_time, end_time)
    
    def fetch_nupl(self, window, start_time=None, end_time=None, sleep_time=0.5):
        print(f"Fetching {self.endpoint_category}/{self.metric} data for {self.currency} with window '{window}'...")

        endpoint = f"{self.currency}/{self.endpoint_category}/{self.metric}"
        remaining_limit = self.limit  # Track how many data points you want in total

        while remaining_limit > 0:
            # Construct params based on allowed combinations
            params = {
                "window": window
            }

            # Always fetch at most 1000 (or remaining)
            current_batch_limit = min(remaining_limit, 1000)
            params["limit"] = current_batch_limit

            if start_time:
                params["start_time"] = start_time
                print(f"Using start_time + limit={current_batch_limit} from {datetime.utcfromtimestamp(start_time/1000)}")
            elif end_time:
                params["end_time"] = end_time
                print(f"Using end_time + limit={current_batch_limit} to {datetime.utcfromtimestamp(end_time/1000)}")
            else:
                raise ValueError("You must provide at least 'start_time' or 'end_time'.")

            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)

            if response.status_code == 200:
                response_json = response.json()
                data = response_json.get("data", [])

                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry["start_time"],
                        "nupl": entry.get("nupl"),  # Adjust as needed
                        "nup": entry.get("nup"),
                        "nul": entry.get("nul")
                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                remaining_limit -= len(data)

                # Prepare for next loop
                last_timestamp = data[-1]["start_time"]
                if end_time and (last_timestamp + 1000 > end_time):
                    break

                start_time = last_timestamp + 1000
                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.metric, window, start_time, end_time)

    def fetch_cdd(self, window, start_time=None, end_time=None, sleep_time=0.5):
        print(f"Fetching {self.endpoint_category}/{self.metric} data for {self.currency} with window '{window}'...")

        endpoint = f"{self.currency}/{self.endpoint_category}/{self.metric}"
        remaining_limit = self.limit  # Track how many data points you want in total

        while remaining_limit > 0:
            # Construct params based on allowed combinations
            params = {
                "window": window
            }

            # Always fetch at most 1000 (or remaining)
            current_batch_limit = min(remaining_limit, 1000)
            params["limit"] = current_batch_limit

            if start_time:
                params["start_time"] = start_time
                print(f"Using start_time + limit={current_batch_limit} from {datetime.utcfromtimestamp(start_time/1000)}")
            elif end_time:
                params["end_time"] = end_time
                print(f"Using end_time + limit={current_batch_limit} to {datetime.utcfromtimestamp(end_time/1000)}")
            else:
                raise ValueError("You must provide at least 'start_time' or 'end_time'.")

            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)

            if response.status_code == 200:
                response_json = response.json()
                data = response_json.get("data", [])

                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry["start_time"],
                        "cdd": entry.get("cdd"),  # Adjust as needed
                        "sa_cdd": entry.get("sa_cdd"),
                        "average_sa_cdd": entry.get("average_sa_cdd"),
                        "binary_cdd": entry.get("binary_cdd")

                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                remaining_limit -= len(data)

                # Prepare for next loop
                last_timestamp = data[-1]["start_time"]
                if end_time and (last_timestamp + 1000 > end_time):
                    break

                start_time = last_timestamp + 1000
                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.metric, window, start_time, end_time)
    
    def fetch_puell_multiple(self, window, start_time=None, end_time=None, sleep_time=0.5):
        print(f"Fetching {self.endpoint_category}/{self.metric} data for {self.currency} with window '{window}'...")

        endpoint = f"{self.currency}/{self.endpoint_category}/{self.metric}"
        remaining_limit = self.limit  # Track how many data points you want in total

        while remaining_limit > 0:
            # Construct params based on allowed combinations
            params = {
                "window": window
            }

            # Always fetch at most 1000 (or remaining)
            current_batch_limit = min(remaining_limit, 1000)
            params["limit"] = current_batch_limit

            if start_time:
                params["start_time"] = start_time
                print(f"Using start_time + limit={current_batch_limit} from {datetime.utcfromtimestamp(start_time/1000)}")
            elif end_time:
                params["end_time"] = end_time
                print(f"Using end_time + limit={current_batch_limit} to {datetime.utcfromtimestamp(end_time/1000)}")
            else:
                raise ValueError("You must provide at least 'start_time' or 'end_time'.")

            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)

            if response.status_code == 200:
                response_json = response.json()
                data = response_json.get("data", [])

                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry["start_time"],
                        "puell_multiple": entry.get("puell_multiple")
                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                remaining_limit -= len(data)

                # Prepare for next loop
                last_timestamp = data[-1]["start_time"]
                if end_time and (last_timestamp + 1000 > end_time):
                    break

                start_time = last_timestamp + 1000
                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.metric, window, start_time, end_time)

    def fetch_hashrate(self, window, start_time=None, end_time=None, sleep_time=0.5):
        print(f"Fetching {self.endpoint_category}/{self.metric} data for {self.currency} with window '{window}'...")

        endpoint = f"{self.currency}/{self.endpoint_category}/{self.metric}"
        remaining_limit = self.limit  # Track how many data points you want in total

        while remaining_limit > 0:
            # Construct params based on allowed combinations
            params = {
                "window": window
            }

            # Always fetch at most 1000 (or remaining)
            current_batch_limit = min(remaining_limit, 1000)
            params["limit"] = current_batch_limit

            if start_time:
                params["start_time"] = start_time
                print(f"Using start_time + limit={current_batch_limit} from {datetime.utcfromtimestamp(start_time/1000)}")
            elif end_time:
                params["end_time"] = end_time
                print(f"Using end_time + limit={current_batch_limit} to {datetime.utcfromtimestamp(end_time/1000)}")
            else:
                raise ValueError("You must provide at least 'start_time' or 'end_time'.")

            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)

            if response.status_code == 200:
                response_json = response.json()
                data = response_json.get("data", [])

                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry["start_time"],
                        "hashrate": entry.get("hashrate")
                    })

                print(f"✅ Retrieved {len(data)} records. Total so far: {len(self.data)}")

                remaining_limit -= len(data)

                # Prepare for next loop
                last_timestamp = data[-1]["start_time"]
                if end_time and (last_timestamp + 1000 > end_time):
                    break

                start_time = last_timestamp + 1000
                time.sleep(sleep_time)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                break

        return self.save_to_csv(self.metric, window, start_time, end_time)

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




    def fetch_active_address(self, interval, start_time=None, end_time=None, sleep_time=0.5):
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

                if not isinstance(data, list):
                    print(f"Unexpected response format: {data}")
                    break

                if not data:
                    print("No more data returned.")
                    break

                for entry in data:
                    self.data.append({
                        "timestamp": entry.get("t") or entry.get("start_time"),  # support both formats
                        "active_address": float(entry["v"]) if isinstance(entry["v"], str) else entry["v"]
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


