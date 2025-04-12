from .base import Strategy
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class NLPStrategy(Strategy):
    '''
    NLP Strategy that uses Natural Language Processing to analyze news sentiment and make trading decisions.
    
    This strategy uses sentiment scores from NLP analysis to generate buy/sell signals:
    - Buy when sentiment improves significantly from negative to positive
    - Sell when sentiment drops significantly from positive to negative
    - Leverage additional metrics like sentiment momentum, volatility, and article count
    '''
    
    def __init__(self, 
                 sentiment_threshold=0.1, 
                 momentum_threshold=0.05,
                 ma_period=3,
                 volume_threshold=0.5,
                 use_article_count=True,
                 combine_price_action=True):
        """
        Initialize the NLP strategy with configurable parameters.
        
        Parameters:
        -----------
        sentiment_threshold : float
            Threshold for sentiment score to trigger signals
        momentum_threshold : float
            Threshold for sentiment momentum to confirm signals
        ma_period : int
            Period for moving average calculation
        volume_threshold : float
            Threshold for volume to confirm signals
        use_article_count : bool
            Whether to factor in article count in decision making
        combine_price_action : bool
            Whether to combine price action with sentiment
        """
        self.sentiment_threshold = sentiment_threshold
        self.momentum_threshold = momentum_threshold
        self.ma_period = ma_period
        self.volume_threshold = volume_threshold
        self.use_article_count = use_article_count
        self.combine_price_action = combine_price_action
        
        # State variables
        self.prev_sentiment = None
        self.prev_price = None
        self.in_position = False
        self.wait_days = 0
        
    def generate_signal(self, df, date, sentiment_data, normalizer=None):
        """
        Generate trading signals based on sentiment data and market conditions.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Market data for the current period
        date : datetime
            Current date
        sentiment_data : dict
            Sentiment data for the current date
        normalizer : ExternalNormalizer, optional
            Normalizer for features
            
        Returns:
        --------
        int
            1 for buy, -1 for sell, 0 for hold
        """
        if sentiment_data is None:
            return 0
        
        # Extract sentiment metrics
        sentiment_mean = sentiment_data.get('sentiment_mean', 0)
        sentiment_momentum = sentiment_data.get('sentiment_momentum', 0)
        sentiment_ma3 = sentiment_data.get('sentiment_ma3', sentiment_mean)
        sentiment_std = sentiment_data.get('sentiment_std', 0)
        article_count = sentiment_data.get('article_count', 0)
        positive_ratio = sentiment_data.get('positive_ratio', 0.5)
        negative_ratio = sentiment_data.get('negative_ratio', 0.5)
        
        # If in cooldown period, decrement wait days and hold
        if self.wait_days > 0:
            self.wait_days -= 1
            return 0
            
        # Default signal is hold
        signal = 0
        
        # Extract price data
        current_price = df['close']
        price_change = 0
        if self.prev_price is not None:
            price_change = (current_price - self.prev_price) / self.prev_price
            
        self.prev_price = current_price
        
        # Consider volume if normalizer is provided
        volume_signal = 0
        if normalizer and hasattr(df, 'volume'):
            try:
                volume_norm = normalizer.normalize(df, 'volume')
                if volume_norm > self.volume_threshold:
                    volume_signal = 1
            except:
                volume_signal = 0
                
        # Calculate sentiment signal
        sentiment_signal = 0
        
        # Strong positive sentiment with momentum
        if sentiment_mean > self.sentiment_threshold and sentiment_momentum > self.momentum_threshold:
            sentiment_signal = 1
            
        # Strong negative sentiment with momentum    
        elif sentiment_mean < -self.sentiment_threshold and sentiment_momentum < -self.momentum_threshold:
            sentiment_signal = -1
            
        # Article count factor - more articles = stronger signal
        article_factor = 1
        if self.use_article_count and article_count > 0:
            # Normalize article count - more articles give stronger weight
            article_factor = min(1.5, 1 + (article_count / 100))
        
        # Combine signals
        if sentiment_signal != 0:
            # Combine with price action if enabled
            if self.combine_price_action:
                # Align price and sentiment direction for stronger signal
                if (sentiment_signal > 0 and price_change > 0) or (sentiment_signal < 0 and price_change < 0):
                    signal = sentiment_signal
                # Conflicting signals - use sentiment with reduced strength
                else:
                    signal = sentiment_signal * 0.5
            else:
                signal = sentiment_signal
                
            # Apply article factor
            signal *= article_factor
            
            # Apply volume confirmation if available
            if volume_signal != 0:
                signal *= 1.2
                
        # Decision rules for final signal
        if signal > 0.7 and not self.in_position:
            # Buy signal
            self.in_position = True
            self.wait_days = 1  # Wait at least 1 day after buying
            return 1
        elif signal < -0.7 and self.in_position:
            # Sell signal
            self.in_position = False
            self.wait_days = 1  # Wait at least 1 day after selling
            return -1
            
        # Update previous sentiment
        self.prev_sentiment = sentiment_mean
        
        return 0
        
    def backtest(self, market_data, sentiment_data, start_date=None, end_date=None):
        """
        Backtest the NLP strategy using historical market and sentiment data.
        
        Parameters:
        -----------
        market_data : pandas.DataFrame
            Historical market data
        sentiment_data : pandas.DataFrame
            Historical sentiment data
        start_date : datetime, optional
            Start date for backtest
        end_date : datetime, optional
            End date for backtest
            
        Returns:
        --------
        pandas.DataFrame
            Backtest results with signals and performance metrics
        """
        # Ensure date columns are datetime
        market_data['date'] = pd.to_datetime(market_data['date'])
        sentiment_data['Date'] = pd.to_datetime(sentiment_data['Date'])
        
        # Filter by date range if provided
        if start_date:
            market_data = market_data[market_data['date'] >= pd.to_datetime(start_date)]
            sentiment_data = sentiment_data[sentiment_data['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            market_data = market_data[market_data['date'] <= pd.to_datetime(end_date)]
            sentiment_data = sentiment_data[sentiment_data['Date'] <= pd.to_datetime(end_date)]
            
        # Initialize results
        results = []
        position = 0  # 0 = no position, 1 = long
        entry_price = 0
        self.in_position = False
        self.wait_days = 0
        
        # Sort data by date
        market_data = market_data.sort_values('date')
        sentiment_data = sentiment_data.sort_values('Date')
        
        # Create sentiment lookup dict for fast access
        sentiment_lookup = {}
        for _, row in sentiment_data.iterrows():
            date_key = row['Date'].date()
            sentiment_lookup[date_key] = row.to_dict()
            
        # Process each market data point
        for _, row in market_data.iterrows():
            date = row['date'].date()
            
            # Get sentiment data for this date
            sentiment = sentiment_lookup.get(date)
            
            # If no sentiment for exact date, look for closest previous date
            if sentiment is None:
                prev_dates = [d for d in sentiment_lookup.keys() if d < date]
                if prev_dates:
                    closest_date = max(prev_dates)
                    sentiment = sentiment_lookup[closest_date]
            
            # Generate signal
            signal = self.generate_signal(row, date, sentiment)
            
            # Execute trades
            pnl = 0
            if signal == 1 and position == 0:  # Buy
                position = 1
                entry_price = row['close']
            elif signal == -1 and position == 1:  # Sell
                position = 0
                pnl = (row['close'] - entry_price) / entry_price
                
            # Record results
            results.append({
                'date': row['date'],
                'close': row['close'],
                'signal': signal,
                'position': position,
                'pnl': pnl,
                'sentiment': sentiment['sentiment_mean'] if sentiment else 0,
                'sentiment_momentum': sentiment['sentiment_momentum'] if sentiment and 'sentiment_momentum' in sentiment else 0,
                'article_count': sentiment['article_count'] if sentiment else 0
            })
            
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate cumulative returns
        results_df['cumulative_pnl'] = results_df['pnl'].cumsum()
        
        return results_df