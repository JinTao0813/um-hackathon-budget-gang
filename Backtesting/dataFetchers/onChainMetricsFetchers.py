from .base import DataFetcher
import requests
import os
import time
from datetime import datetime
import pandas as pd


class CryptoQuantFetcher(DataFetcher):
    def __init__(self, api_key, base_url, currency, endpoint_category, metric, exchange, limit=1000):
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
        remaining_limit = self.limit  # Track how many data points you want in total

        while remaining_limit > 0:
            # Construct params based on allowed combinations
            params = {
                "window": window,
                "exchange": self.exchange
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
                        "netflow_total": entry.get("netflow_total")  # Adjust as needed
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

    def save_to_csv(self, metric, window, start_time, end_time):
        df = self.to_dataframe()
        os.makedirs("datasets", exist_ok=True)
        csv_path = f"datasets/{metric}_{window}_Training data_{start_time}_to_{end_time}.csv"
        self.saved_filepath = csv_path
        df.to_csv(csv_path)
        print(f"💾 Saved to {csv_path} with {len(df)} rows.")
        print(df.tail())