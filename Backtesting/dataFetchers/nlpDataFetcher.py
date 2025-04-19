from .base import DataFetcher
import pandas as pd
import os
import glob
from datetime import datetime

# Fetcher for NLP sentiment data
class NLPSentimentFetcher(DataFetcher):
    """
    Fetcher for NLP sentiment data exported from nlp_app.py.
    Loads sentiment data from CSV files and prepares it for use in backtesting.
    """
    def __init__(self, sentiment_file_path=None, sentiment_dir=None):
        """
        Initialize the NLP sentiment data fetcher.
        
        Parameters:
        -----------
        sentiment_file_path : str, optional
            Path to a specific sentiment CSV file to load
        sentiment_dir : str, optional
            Directory containing sentiment CSV files. If provided, the most recent file will be used
        """
        self.sentiment_file_path = sentiment_file_path # Specific file path
        self.sentiment_dir = sentiment_dir # Directory to search for sentiment files
        self.data = [] # List to store sentiment data
        self.sentiment_df = None # DataFrame to store loaded sentiment data
        
    def fetch(self):
        """
        Fetch sentiment data from the specified CSV file or most recent file in directory.
        """
        file_path = self._get_sentiment_file_path()
        if not file_path:
            print("❌ No sentiment data file found.")
            return None
            
        try:
            print(f"📊 Loading sentiment data from: {file_path}")
            self.sentiment_df = pd.read_csv(file_path)
            
            # Ensure date column is properly formatted
            self.sentiment_df['Date'] = pd.to_datetime(self.sentiment_df['Date'])
            
            # Convert data to list of dictionaries for consistency with other fetchers
            for _, row in self.sentiment_df.iterrows():
                self.data.append(dict(row))
                
            print(f"✅ Loaded {len(self.sentiment_df)} days of sentiment data")
            return self.to_dataframe()
            
        except Exception as e:
            print(f"❌ Error loading sentiment data: {str(e)}")
            return None
    
    def _get_sentiment_file_path(self):
        """
        Determine which sentiment file to load based on initialization parameters.
        """
        # If a specific file is provided, use it
        if self.sentiment_file_path and os.path.exists(self.sentiment_file_path): 
            return self.sentiment_file_path
            
        # If a directory is provided, find the most recent sentiment file
        if self.sentiment_dir and os.path.isdir(self.sentiment_dir):
            sentiment_files = glob.glob(os.path.join(self.sentiment_dir, "sentiment_data_*.csv")) # Search for sentiment files in the directory
            if sentiment_files:
                # Sort by modification time, most recent first
                most_recent_file = max(sentiment_files, key=os.path.getmtime)
                return most_recent_file
                
        # Try to find sentiment files in the project root
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sentiment_files = glob.glob(os.path.join(root_dir, "sentiment_data_*.csv"))
        if sentiment_files:
            most_recent_file = max(sentiment_files, key=os.path.getmtime) # Sort by modification time, most recent first
            return most_recent_file
            
        return None
    
    def to_dataframe(self):
        """
        Convert the data to a pandas DataFrame.
        """
        if self.sentiment_df is not None:
            return self.sentiment_df
        return pd.DataFrame(self.data)
        
    def get_sentiment_for_date(self, target_date):
        """
        Get sentiment data for a specific date.
        
        Parameters:
        -----------
        target_date : datetime or str
            The date to get sentiment data for
            
        Returns:
        --------
        dict or None
            Dictionary with sentiment features for the specified date, or None if not found
        """
        if self.sentiment_df is None: ## If sentiment data is not loaded, fetch it
            self.fetch()
            
        if self.sentiment_df is None: # If still not loaded, return None
            return None
            
        # Convert target_date to pandas datetime if it's a string
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
            
        # Find the row for the target date
        date_mask = self.sentiment_df['Date'].dt.date == target_date.date()
        matching_rows = self.sentiment_df[date_mask]
        
        if not matching_rows.empty:
            return matching_rows.iloc[0].to_dict()
        
        # If no exact match, find the closest previous date
        previous_dates = self.sentiment_df[self.sentiment_df['Date'] < target_date]
        if not previous_dates.empty:
            closest_row = previous_dates.iloc[previous_dates['Date'].argmax()] # Get the most recent date before target_date
            return closest_row.to_dict()
            
        return None
