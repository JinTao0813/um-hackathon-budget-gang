import pandas as pd
from abc import ABC, abstractmethod

# Abstract class for data fetchers
class DataFetcher(ABC):
    def __init__(self, api_key, base_url, limit=1000): # default limit set to 1000
        self.api_key = api_key
        self.base_url = base_url
        self.limit = limit
        self.headers = {"X-API-KEY": self.api_key}
        self.data = []

    @abstractmethod
    def fetch(self, *args, **kwargs):
        pass

    def to_dataframe(self): # Convert the fetched data to a pandas DataFrame
        df = pd.DataFrame(self.data)
        if 'timestamp' in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
            df.set_index("timestamp", inplace=True)
        return df

    @abstractmethod
    def save_to_csv(self, *args, **kwargs): # Save the fetched data to a CSV file
        pass