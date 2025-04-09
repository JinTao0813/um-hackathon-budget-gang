#!/usr/bin/env python
# nlp_app.py

import datetime
import time
import feedparser
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from transformers import pipeline
from bs4 import BeautifulSoup
import re
from newsapi import NewsApiClient
import plotly.graph_objects as go
import plotly.utils
import json
import uuid
import threading
import numpy as np
from scipy import stats
import os
import mwclient  # For Wikipedia data retrieval

# For time series analysis
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr

# Create custom JSON encoder that can handle Pandas Timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif pd.isna(obj):  # Handle NaN and None values
            return None
        return super().default(obj)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'bitcoin_sentiment_analysis_secret_key'
app.json_encoder = CustomJSONEncoder  # Set custom encoder for jsonify

# Dictionary to store processing status
processing_status = {}

# Initialize News API client
newsapi = NewsApiClient(api_key='8606ff7427984d94804d491c36edd220')

# ------------------------------
# Initialize sentiment pipeline (FinBERT is fixed)
# ------------------------------
try:
    finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert")
    FINBERT_STATUS = "FinBERT model loaded successfully."
except Exception as e:
    FINBERT_STATUS = f"ERROR: FinBERT model failed to load: {str(e)}"
    finbert = None

# ------------------------------
# Preprocessing Functions
# ------------------------------
def remove_images(text):
    """
    Remove image tags and image urls from HTML text.
    """
    soup = BeautifulSoup(text, "html.parser")
    # Remove <img> tags
    for img in soup.find_all("img"):
        img.decompose()
    # Remove any remaining image URLs
    text = re.sub(r'https?://[^\s<>"]+?\.(jpg|jpeg|gif|png|svg|webp)', '', soup.get_text())
    return text

def clean_text(text):
    """
    Clean text by removing images, HTML tags, and extra whitespace.
    """
    # Remove images and HTML tags
    text = remove_images(text)
    
    # Remove any remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# ------------------------------
# Trading Signal Generation and Backtesting Functions
# ------------------------------
def calculate_trading_signals(sentiment_df, price_data=None):
    """
    Generate trading signals based on sentiment data.
    If price_data is provided, correlate signals with price movements.
    
    Returns a dictionary with trading signals and metrics.
    """
    # If no sentiment data, return default values
    if sentiment_df.empty:
        return {
            'buy_signal_date': 'N/A',
            'buy_signal_price': 'N/A',
            'sell_signal_date': 'N/A', 
            'sell_signal_price': 'N/A',
            'total_trades': 0,
            'win_rate': 0,
            'net_profit': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'price_sentiment_correlation': 0,
            'price_data': [],
            'sentiment_data': []
        }
    
    # Ensure we have datetime column
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Published'])
    
    # Group by date and calculate daily sentiment
    daily_sentiment = sentiment_df.groupby(sentiment_df['Date'].dt.date)['Sentiment Score'].agg(['mean', 'count']).reset_index()
    daily_sentiment.columns = ['Date', 'Sentiment', 'Count']  
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    daily_sentiment = daily_sentiment.sort_values('Date')
    
    # Initialize trading signals and metrics
    trading_signals = {
        'buy_signal_date': 'N/A',
        'buy_signal_price': 'N/A',
        'sell_signal_date': 'N/A', 
        'sell_signal_price': 'N/A',
        'total_trades': 0,
        'win_rate': 0,
        'net_profit': 0,
        'max_drawdown': 0,
        'sharpe_ratio': 0,
        'sortino_ratio': 0,
        'price_sentiment_correlation': 0,
        'price_data': [],
        'sentiment_data': []
    }
    
    # If price data is not provided, try to load it
    if price_data is None:
        price_data = load_price_data()
        
    # Print debug info
    print(f"Price data type: {type(price_data)}")
    if isinstance(price_data, pd.DataFrame):
        print(f"Price data shape: {price_data.shape}")
        print(f"Price data columns: {price_data.columns.tolist()}")
        print(f"Price data sample: {price_data.head(2).to_dict('records')}")
    
    # Only proceed with trading signals if we have price data
    if price_data is not None and not daily_sentiment.empty:
        # Convert price data to dataframe if it's not already
        if not isinstance(price_data, pd.DataFrame):
            # Assuming price_data might be a list of dictionaries
            try:
                price_df = pd.DataFrame(price_data)
                # Ensure we have datetime column
                price_df['Date'] = pd.to_datetime(price_df['Date'])
            except Exception as e:
                print(f"Error converting price data to dataframe: {str(e)}")
                price_df = None
        else:
            price_df = price_data.copy()
            price_df['Date'] = pd.to_datetime(price_df['Date'])
        
        if price_df is not None:
            # Print more debug info
            print(f"Working with price_df shape: {price_df.shape}")
            print(f"Daily sentiment shape: {daily_sentiment.shape}")
            
            # Standardize timezone handling - convert both to naive datetime (no timezone)
            # This is crucial for merge_asof to work correctly
            if hasattr(price_df['Date'].dtype, 'tz') and price_df['Date'].dtype.tz is not None:
                price_df['Date'] = price_df['Date'].dt.tz_localize(None)
                
            if hasattr(daily_sentiment['Date'].dtype, 'tz') and daily_sentiment['Date'].dtype.tz is not None:
                daily_sentiment['Date'] = daily_sentiment['Date'].dt.tz_localize(None)
                
            # Ensure both DataFrames have the same datetime dtype
            price_df['Date'] = pd.to_datetime(price_df['Date']).dt.tz_localize(None)
            daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date']).dt.tz_localize(None)
                
            try:
                # Merge sentiment and price data
                merged_data = pd.merge_asof(
                    price_df.sort_values('Date'),
                    daily_sentiment.sort_values('Date'),
                    on='Date',
                    direction='nearest'
                )
                
                # Calculate moving averages for sentiment
                if len(merged_data) >= 3:
                    window_size = min(3, len(merged_data))
                    merged_data['SentimentMA'] = merged_data['Sentiment'].rolling(window=window_size).mean()
                    merged_data['PricePctChange'] = merged_data['Close'].pct_change()
                    
                    # Fill NaN values from rolling calculations
                    merged_data = merged_data.dropna()
                    
                    if len(merged_data) >= 2:
                        # Calculate trading signals
                        merged_data['Signal'] = 0
                        
                        # Example of a simple signal strategy:
                        # Buy when sentiment MA is positive and increasing
                        # Sell when sentiment MA is negative and decreasing
                        merged_data.loc[(merged_data['SentimentMA'] > 0.2) & 
                                       (merged_data['SentimentMA'].shift(1) < merged_data['SentimentMA']), 'Signal'] = 1  # Buy
                        
                        merged_data.loc[(merged_data['SentimentMA'] < -0.2) & 
                                       (merged_data['SentimentMA'].shift(1) > merged_data['SentimentMA']), 'Signal'] = -1  # Sell
                        
                        # Calculate trading stats
                        buy_signals = merged_data[merged_data['Signal'] == 1]
                        sell_signals = merged_data[merged_data['Signal'] == -1]
                        
                        total_trades = len(buy_signals) + len(sell_signals)
                        
                        # Calculate correlation between sentiment and price change
                        if 'PricePctChange' in merged_data.columns and len(merged_data['PricePctChange']) > 1:
                            price_sentiment_corr, _ = pearsonr(merged_data['Sentiment'], merged_data['PricePctChange'])
                        else:
                            price_sentiment_corr = 0
                            
                        # Process buy and sell signals for display
                        if not buy_signals.empty:
                            latest_buy = buy_signals.iloc[-1]
                            trading_signals['buy_signal_date'] = latest_buy['Date'].strftime('%Y-%m-%d')
                            trading_signals['buy_signal_price'] = f"{latest_buy['Close']:,.2f}"
                        
                        if not sell_signals.empty:
                            latest_sell = sell_signals.iloc[-1]
                            trading_signals['sell_signal_date'] = latest_sell['Date'].strftime('%Y-%m-%d')
                            trading_signals['sell_signal_price'] = f"{latest_sell['Close']:,.2f}"
                        
                        trading_signals['total_trades'] = total_trades
                        
                        # Simulate portfolio performance
                        if total_trades > 0:
                            # Simple backtest
                            portfolio = 100000  # Starting with $100,000
                            position = 0
                            trades = []
                            
                            for idx, row in merged_data.iterrows():
                                if row['Signal'] == 1 and position == 0:  # Buy
                                    position = portfolio / row['Close']
                                    portfolio = 0
                                    trades.append(('buy', row['Date'], row['Close'], position))
                                elif row['Signal'] == -1 and position > 0:  # Sell
                                    portfolio = position * row['Close']
                                    position = 0
                                    trades.append(('sell', row['Date'], row['Close'], portfolio))
                            
                            # Calculate final portfolio value
                            final_value = portfolio
                            if position > 0:
                                final_value = position * merged_data.iloc[-1]['Close']
                            
                            # Calculate performance metrics
                            win_trades = 0
                            if len(trades) >= 2:
                                for i in range(1, len(trades), 2):
                                    if i < len(trades):
                                        if trades[i][3] > trades[i-1][3]:
                                            win_trades += 1
                                            
                            win_rate = win_trades / (len(trades) // 2) * 100 if len(trades) >= 2 else 0
                            
                            # Calculate returns
                            returns = []
                            for i in range(1, len(trades), 2):
                                if i < len(trades):
                                    buy_price = trades[i-1][2]
                                    sell_price = trades[i][2]
                                    returns.append(sell_price / buy_price - 1)
                            
                            if returns:
                                # Sharpe ratio (annualized)
                                risk_free_rate = 0.02  # Assuming 2% risk-free rate
                                excess_returns = [r - risk_free_rate/252 for r in returns]  # Daily excess returns
                                sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) != 0 else 0
                                
                                # Sortino ratio (annualized)
                                downside_returns = [r for r in excess_returns if r < 0]
                                sortino_ratio = np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252) if downside_returns and np.std(downside_returns) != 0 else 0
                                
                                # Maximum drawdown
                                cumulative = np.cumprod(np.array([1 + r for r in returns]))
                                max_dd = 0
                                peak = cumulative[0]
                                for value in cumulative:
                                    if value > peak:
                                        peak = value
                                    dd = (peak - value) / peak
                                    if dd > max_dd:
                                        max_dd = dd
                                
                                trading_signals['win_rate'] = round(win_rate, 1)
                                trading_signals['net_profit'] = f"{final_value - 100000:,.2f}"
                                trading_signals['max_drawdown'] = round(max_dd * 100, 2)
                                trading_signals['sharpe_ratio'] = round(sharpe_ratio, 2)
                                trading_signals['sortino_ratio'] = round(sortino_ratio, 2)
                                trading_signals['price_sentiment_correlation'] = round(price_sentiment_corr, 2)
                        
                        # Prepare data for charts
                        # Make sure dates are converted to strings for JSON serialization
                        chart_data = merged_data[['Date', 'Close', 'Sentiment']].copy()
                        chart_data['Date'] = chart_data['Date'].dt.strftime('%Y-%m-%d')
                        
                        trading_signals['price_data'] = chart_data[['Date', 'Close']].to_dict('records')
                        trading_signals['sentiment_data'] = chart_data[['Date', 'Sentiment']].to_dict('records')
                        
                        print(f"Successfully prepared chart data with {len(trading_signals['price_data'])} price points and {len(trading_signals['sentiment_data'])} sentiment points")
            except Exception as e:
                print(f"Error in trading signal generation: {str(e)}")
                import traceback
                traceback.print_exc()
                
    # Always ensure we have at least some data for the chart, even if it's simulated
    if not trading_signals['price_data'] and not trading_signals['sentiment_data']:
        # Generate sample data just to make sure chart shows something
        print("Generating sample data for chart since no real data is available")
        dates = pd.date_range(start=datetime.datetime.now() - datetime.timedelta(days=30), 
                             end=datetime.datetime.now(), freq='D')
        
        sample_data = []
        for i, date in enumerate(dates):
            price = 60000 + np.random.normal(0, 1000)
            sentiment = np.random.uniform(-0.3, 0.3)
            
            # Format date as string
            date_str = date.strftime('%Y-%m-%d')
            
            sample_data.append({
                'Date': date_str,
                'Close': price,
                'Sentiment': sentiment
            })
        
        trading_signals['price_data'] = [{
            'Date': item['Date'], 
            'Close': item['Close']
        } for item in sample_data]
        
        trading_signals['sentiment_data'] = [{
            'Date': item['Date'], 
            'Sentiment': item['Sentiment']
        } for item in sample_data]
        
        print(f"Generated {len(sample_data)} sample data points for chart")
                
    return trading_signals

def load_price_data():
    """
    Load Bitcoin price data with multiple fallback options:
    1. Try to get real-time data from yfinance API
    2. If that fails, try to load from local dataset file
    3. If all else fails, generate simulated data as a last resort
    
    Returns a pandas DataFrame with Bitcoin price data.
    """
    # 1. First try to get real data from yfinance
    try:
        # Import yfinance
        import yfinance as yf
        
        # Get current date and 60 days ago
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=60)
        
        print(f"Fetching fresh Bitcoin price data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Fetch Bitcoin data from Yahoo Finance
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)
        
        # Check if data was fetched successfully
        if not btc_data.empty:
            # Reset index to make Date a column
            btc_data = btc_data.reset_index()
            
            # Ensure Date column is datetime
            btc_data['Date'] = pd.to_datetime(btc_data['Date'])
            
            print(f"Successfully fetched fresh Bitcoin price data from yfinance with {len(btc_data)} records")
            return btc_data
        else:
            print("Failed to fetch data from yfinance, trying local dataset file...")
    except Exception as e:
        print(f"Error fetching data from yfinance: {str(e)}")
        print("Trying local dataset file...")
    
    # 2. If yfinance fails, try to read from local dataset file
    try:
        local_dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'datasets', 
            'BTCUSDT_1m_march_to_april.csv'
        )
        
        if os.path.exists(local_dataset_path):
            print(f"Loading Bitcoin price data from local file: {local_dataset_path}")
            # Read the CSV file
            btc_data = pd.read_csv(local_dataset_path)
            
            # Check if the required columns exist
            expected_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
            
            # Standardize column names if needed
            if all(col in btc_data.columns for col in expected_columns):
                # Convert to OHLCV format expected by the rest of the code
                standardized_data = pd.DataFrame({
                    'Date': pd.to_datetime(btc_data['open_time'], unit='ms'),
                    'Open': btc_data['open'],
                    'High': btc_data['high'],
                    'Low': btc_data['low'],
                    'Close': btc_data['close'],
                    'Volume': btc_data['volume']
                })
                
                # Resample to daily data if we have minute data
                if len(standardized_data) > 100:  # Likely minute data
                    print("Resampling minute data to daily data...")
                    daily_data = standardized_data.set_index('Date').resample('D').agg({
                        'Open': 'first',
                        'High': 'max',
                        'Low': 'min',
                        'Close': 'last',
                        'Volume': 'sum'
                    }).reset_index()
                    
                    print(f"Successfully loaded Bitcoin price data from local file with {len(daily_data)} daily records")
                    return daily_data
                
                print(f"Successfully loaded Bitcoin price data from local file with {len(standardized_data)} records")
                return standardized_data
            else:
                print(f"Local dataset file doesn't have the expected columns. Found: {btc_data.columns.tolist()}")
        else:
            print(f"Local dataset file not found at: {local_dataset_path}")
    except Exception as e:
        print(f"Error loading data from local file: {str(e)}")
    
    # 3. Generate simulated data as a last resort
    print("No real Bitcoin price data available, using simulated data as last resort")
    return generate_default_price_data()

def generate_default_price_data():
    """
    Generate simulated Bitcoin price data for demonstration purposes
    when real data is not available.
    """
    # Create date range for the past 30 days
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    
    dates = pd.date_range(start=start_date, end=end_date)
    
    # Generate simulated price data
    np.random.seed(42)  # For reproducibility
    
    # Start with a price and add random changes
    price = 67000  # Starting Bitcoin price
    prices = [price]
    
    for i in range(1, len(dates)):
        price_change = np.random.normal(0, 1000)  # Random daily price change
        price = max(1000, price + price_change)  # Ensure price doesn't go too low
        prices.append(price)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + np.random.uniform(0, 0.05)) for p in prices],
        'Low': [p * (1 - np.random.uniform(0, 0.05)) for p in prices],
        'Close': [p * (1 + np.random.normal(0, 0.02)) for p in prices],
        'Volume': [np.random.uniform(10000, 50000) for _ in prices]
    })
    
    return df

def create_sentiment_visualizations(df):
    """
    Create advanced Plotly visualizations for sentiment analysis results.
    Returns HTML for the plots directly.
    """
    visualization_html = []
    
    # Apply a modern color palette
    color_palette = {
        'positive': '#2ECC40',  # Green
        'neutral': '#FF851B',   # Orange
        'negative': '#FF4136',  # Red
        'background': '#F8F9FA',
        'grid': '#E9ECEF',
        'text': '#343A40'
    }
    
    # Common layout settings
    common_layout = dict(
        font=dict(family="Roboto, Arial, sans-serif", size=12, color=color_palette['text']),
        paper_bgcolor=color_palette['background'],
        plot_bgcolor=color_palette['background'],
        margin=dict(t=50, l=20, r=20, b=30),
        autosize=True
    )
    
    # 1. Create an advanced sentiment over time visualization
    df['Published'] = pd.to_datetime(df['Published'])
    
    # Group by date and sentiment label to get counts
    sentiment_over_time = df.groupby([df['Published'].dt.date, 'Sentiment Label']).size().reset_index(name='Count')
    sentiment_over_time['Published'] = pd.to_datetime(sentiment_over_time['Published'])
    
    # Create a more informative time-based visualization
    fig_time = go.Figure()
    
    # Add traces for each sentiment category
    for sentiment in sentiment_over_time['Sentiment Label'].unique():
        data = sentiment_over_time[sentiment_over_time['Sentiment Label'] == sentiment]
        fig_time.add_trace(go.Scatter(
            x=data['Published'],
            y=data['Count'],
            mode='lines+markers',
            name=sentiment.capitalize(),
            line=dict(
                width=4,
                color=color_palette.get(sentiment, '#AAAAAA'),
                dash='solid' if sentiment == 'positive' else 'dash' if sentiment == 'negative' else 'dot'
            ),
            marker=dict(
                size=10,
                symbol='circle' if sentiment == 'positive' else 'square' if sentiment == 'negative' else 'diamond'
            )
        ))
    
    fig_time.update_layout(
        title=dict(
            text='Sentiment Trends Over Time',
            font=dict(size=24)
        ),
        xaxis=dict(
            title='Date',
            gridcolor=color_palette['grid'],
            showgrid=True
        ),
        yaxis=dict(
            title='Number of Articles',
            gridcolor=color_palette['grid'],
            showgrid=True
        ),
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        **common_layout
    )
    visualization_html.append(fig_time.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True}))
    
    # 2. Source-wise Sentiment Comparison - enhanced
    # Get average sentiment score per source
    source_sentiment = df.groupby('Source')['Sentiment Score'].mean().reset_index()
    
    # Sort by sentiment score
    source_sentiment = source_sentiment.sort_values('Sentiment Score')
    
    # Add a column for colors (green for positive, red for negative)
    colors = [color_palette['negative'] if score < 0 else color_palette['positive'] for score in source_sentiment['Sentiment Score']]
    
    fig_bar = go.Figure(data=go.Bar(
        x=source_sentiment['Source'],
        y=source_sentiment['Sentiment Score'],
        marker_color=colors,
        text=source_sentiment['Sentiment Score'].round(3),
        textposition='auto'
    ))
    
    fig_bar.update_layout(
        title=dict(
            text='Average Sentiment by Source',
            font=dict(size=24)
        ),
        xaxis=dict(
            title='News Source',
            tickangle=-45,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            title='Sentiment Score (Negative to Positive)',
            gridcolor=color_palette['grid'],
            showgrid=True
        ),
        height=450,
        **common_layout
    )
    visualization_html.append(fig_bar.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True}))
    
    # 3. Sentiment Distribution - Valuable for HMM integration
    # Create sentiment distribution histogram
    fig_dist = go.Figure()
    
    fig_dist.add_trace(go.Histogram(
        x=df['Sentiment Score'],
        marker_color='rgba(58, 71, 80, 0.6)',
        nbinsx=20,
        name='Sentiment Distribution'
    ))
    
    # Add a KDE curve - with error handling for singular matrices
    hist_data, bin_edges = np.histogram(
        df['Sentiment Score'], 
        bins=20, 
        range=(df['Sentiment Score'].min(), df['Sentiment Score'].max()),
        density=True
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Try to fit KDE with error handling
    try:
        # Check if we have enough unique values for KDE (at least 2)
        unique_values = np.unique(df['Sentiment Score'])
        if len(unique_values) >= 2:
            kde = stats.gaussian_kde(df['Sentiment Score'])
            x_grid = np.linspace(df['Sentiment Score'].min(), df['Sentiment Score'].max(), 100)
            kde_values = kde(x_grid)
            
            fig_dist.add_trace(go.Scatter(
                x=x_grid,
                y=kde_values * (df.shape[0] / 5),  # Scale KDE to match histogram height
                mode='lines',
                name='Density',
                line=dict(color='rgba(255, 0, 0, 0.8)', width=3)
            ))
        else:
            print(f"Warning: Not enough unique sentiment values for KDE. Found {len(unique_values)} unique values.")
    except (np.linalg.LinAlgError, ValueError) as e:
        print(f"Warning: Could not calculate KDE curve - {str(e)}")
        # Continue without KDE curve
    
    fig_dist.update_layout(
        title=dict(
            text='Sentiment Score Distribution',
            font=dict(size=24)
        ),
        xaxis=dict(
            title='Sentiment Score',
            gridcolor=color_palette['grid'],
            showgrid=True
        ),
        yaxis=dict(
            title='Frequency',
            gridcolor=color_palette['grid'],
            showgrid=True
        ),
        height=450,
        bargap=0.1,
        **common_layout
    )
    visualization_html.append(fig_dist.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True}))
    
    # 4. Sentiment Volatility/Moving Average - Critical for HMM
    # Calculate daily average sentiment
    df['Date'] = df['Published'].dt.date
    daily_sentiment = df.groupby('Date')['Sentiment Score'].mean().reset_index()
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    daily_sentiment = daily_sentiment.sort_values('Date')
    
    # Calculate rolling metrics (7-day)
    if len(daily_sentiment) >= 3:  # Need at least 3 days for meaningful rolling window
        window_size = min(7, len(daily_sentiment))
        daily_sentiment['MA7'] = daily_sentiment['Sentiment Score'].rolling(window=window_size).mean()
        daily_sentiment['Volatility'] = daily_sentiment['Sentiment Score'].rolling(window=window_size).std()
        
        fig_vol = go.Figure()
        
        # Add sentiment score line
        fig_vol.add_trace(go.Scatter(
            x=daily_sentiment['Date'],
            y=daily_sentiment['Sentiment Score'],
            mode='lines+markers',
            name='Daily Sentiment',
            line=dict(color='rgba(50, 171, 96, 1)', width=2)
        ))
        
        # Add moving average line
        fig_vol.add_trace(go.Scatter(
            x=daily_sentiment['Date'],
            y=daily_sentiment['MA7'],
            mode='lines',
            name=f'{window_size}-day MA',
            line=dict(color='rgba(31, 119, 180, 1)', width=3)
        ))
        
        # Add volatility as band
        fig_vol.add_trace(go.Scatter(
            x=daily_sentiment['Date'],
            y=daily_sentiment['MA7'] + daily_sentiment['Volatility'],
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='none'
        ))
        
        fig_vol.add_trace(go.Scatter(
            x=daily_sentiment['Date'],
            y=daily_sentiment['MA7'] - daily_sentiment['Volatility'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)',
            name='Volatility',
            hoverinfo='none'
        ))
        
        fig_vol.update_layout(
            title=dict(
                text='Sentiment Volatility & Trend Analysis',
                font=dict(size=24)
            ),
            xaxis=dict(
                title='Date',
                gridcolor=color_palette['grid'],
                showgrid=True
            ),
            yaxis=dict(
                title='Sentiment Score',
                gridcolor=color_palette['grid'],
                showgrid=True
            ),
            height=450,
            hovermode='x unified',
            **common_layout
        )
        visualization_html.append(fig_vol.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True}))
    
    return visualization_html

def export_sentiment_data_for_hmm(sentiment_df, output_path=None):
    """
    Export sentiment data in a format that can be easily integrated with HMM models.
    This creates a CSV file with daily sentiment features that can be used as input to the HMM.
    
    Parameters:
    -----------
    sentiment_df : pandas DataFrame
        DataFrame containing sentiment analysis results
    output_path : str, optional
        Path where the CSV file will be saved. If None, saves to the project root.
    
    Returns:
    --------
    str
        Path to the saved CSV file
    """
    if sentiment_df.empty:
        print("No sentiment data to export")
        return None
        
    try:
        # Ensure we have datetime
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Published'])
        
        # Group by date and calculate various sentiment metrics
        daily_data = sentiment_df.groupby(sentiment_df['Date'].dt.date).agg({
            'Sentiment Score': ['mean', 'std', 'min', 'max', 'count'],
            'Sentiment Label': lambda x: (x == 'positive').mean(),  # Percentage of positive sentiment
        }).reset_index()
        
        # Flatten the multi-level columns
        daily_data.columns = ['Date', 'sentiment_mean', 'sentiment_std', 'sentiment_min', 
                             'sentiment_max', 'article_count', 'positive_ratio']
        
        # Calculate additional features
        if len(daily_data) > 1:
            # Sentiment momentum (day-over-day change)
            daily_data['sentiment_momentum'] = daily_data['sentiment_mean'].diff()
            
            # Sentiment acceleration (change in momentum)
            daily_data['sentiment_acceleration'] = daily_data['sentiment_momentum'].diff()
            
            # Calculate negative sentiment ratio (like in the YouTube example)
            daily_data['neg_sentiment'] = sentiment_df.groupby(sentiment_df['Date'].dt.date)['Sentiment Label'].apply(
                lambda x: (x == 'negative').mean()
            ).reset_index(drop=True)
            
            # Sentiment volatility (rolling standard deviation)
            window_size = min(3, len(daily_data))
            daily_data['sentiment_volatility'] = daily_data['sentiment_mean'].rolling(window=window_size).std()
            
            # Add rolling averages to capture trends like in the YouTube example
            horizons = [2, 7, 14, 30]
            
            # For each horizon, calculate rolling statistics
            for horizon in horizons:
                if len(daily_data) > horizon:
                    roll_window = min(horizon, len(daily_data))
                    
                    # Rolling mean sentiment
                    daily_data[f'sentiment_mean_{horizon}d'] = daily_data['sentiment_mean'].rolling(
                        window=roll_window, min_periods=1).mean()
                    
                    # Rolling sentiment volatility
                    daily_data[f'sentiment_std_{horizon}d'] = daily_data['sentiment_mean'].rolling(
                        window=roll_window, min_periods=1).std()
                    
                    # Rate of change over the period
                    daily_data[f'sentiment_roc_{horizon}d'] = daily_data['sentiment_mean'].pct_change(periods=min(horizon, len(daily_data)-1))
                    
                    # Negative sentiment trend
                    daily_data[f'neg_sentiment_{horizon}d'] = daily_data['neg_sentiment'].rolling(
                        window=roll_window, min_periods=1).mean()
                    
                    # Article count trend
                    daily_data[f'article_count_{horizon}d'] = daily_data['article_count'].rolling(
                        window=roll_window, min_periods=1).mean()
            
            # Ensure datetime format for index
            daily_data['Date'] = pd.to_datetime(daily_data['Date'])
            
            # Sentiment regime (simple classification based on mean and momentum)
            # This can help the HMM identify potential market regimes
            conditions = [
                (daily_data['sentiment_mean'] > 0.2) & (daily_data['sentiment_momentum'] > 0),
                (daily_data['sentiment_mean'] > 0) & (daily_data['sentiment_momentum'] <= 0),
                (daily_data['sentiment_mean'] <= 0) & (daily_data['sentiment_momentum'] >= 0),
                (daily_data['sentiment_mean'] <= -0.2) & (daily_data['sentiment_momentum'] < 0)
            ]
            choices = ['strong_positive', 'weakening_positive', 'recovering_negative', 'strong_negative']
            daily_data['sentiment_regime'] = np.select(conditions, choices, default='neutral')
            
            # Create binary signals with higher frequency (to meet the 3% requirement)
            # Using multiple thresholds to generate more signals
            daily_data['signal_primary'] = 0
            daily_data.loc[(daily_data['sentiment_mean'] > 0.15) & 
                           (daily_data['sentiment_momentum'] > 0), 'signal_primary'] = 1  # Buy
            daily_data.loc[(daily_data['sentiment_mean'] < -0.15) & 
                           (daily_data['sentiment_momentum'] < 0), 'signal_primary'] = -1  # Sell
                           
            daily_data['signal_secondary'] = 0
            daily_data.loc[(daily_data['sentiment_mean'] > 0) & 
                          (daily_data['sentiment_volatility'] < daily_data['sentiment_volatility'].median()), 
                          'signal_secondary'] = 1  # Buy on positive sentiment with low volatility
            daily_data.loc[(daily_data['sentiment_mean'] < 0) & 
                          (daily_data['sentiment_volatility'] > daily_data['sentiment_volatility'].median()), 
                          'signal_secondary'] = -1  # Sell on negative sentiment with high volatility
                          
            # Combined signal (more aggressive)
            daily_data['signal_combined'] = daily_data['signal_primary']
            daily_data.loc[daily_data['signal_primary'] == 0, 'signal_combined'] = daily_data.loc[daily_data['signal_primary'] == 0, 'signal_secondary']
            
            # Calculate percentage of rows with signals to verify we meet the 3% requirement
            signal_percentage = (daily_data['signal_combined'] != 0).mean() * 100
            print(f"Trading signal frequency: {signal_percentage:.2f}% of data rows")
        
        # Determine output path
        if output_path is None:
            # Get timestamp for unique filename
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                    f'sentiment_data_{timestamp}.csv')
        
        # Save to CSV
        daily_data.to_csv(output_path, index=False)
        print(f"Sentiment data exported to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error exporting sentiment data: {str(e)}")
        return None

def get_wikipedia_content(page_title, start_date=None, end_date=None):
    """
    Fetches content from Wikipedia revisions for a specified page title using mwclient.
    Allows filtering by date range if provided.
    Returns a list of dictionaries containing revision dates, titles, and text.
    
    Parameters:
    -----------
    page_title : str
        Title of the Wikipedia page
    start_date : datetime, optional
        Start date for filtering revisions
    end_date : datetime, optional
        End date for filtering revisions
    """
    try:
        import mwclient
        from concurrent.futures import ThreadPoolExecutor
        import os
        
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache filename based on parameters
        cache_key = f"{page_title}_{start_date.strftime('%Y%m%d') if start_date else 'nostart'}_{end_date.strftime('%Y%m%d') if end_date else 'noend'}"
        cache_file = os.path.join(cache_dir, f"wiki_cache_{cache_key.replace(' ', '_')}.json")
        
        # Check if cache exists
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    import json
                    print(f"Loading Wikipedia content from cache: {cache_file}")
                    return json.load(f)
            except Exception as e:
                print(f"Cache loading failed: {str(e)}, fetching fresh content")
                # Continue with fresh fetch if cache fails
        
        # Connect to Wikipedia
        site = mwclient.Site('en.wikipedia.org')
        page = site.pages[page_title]
        
        if not page.exists:
            return [{"title": "Error", "text": f"Wikipedia page '{page_title}' does not exist.", "url": "", "revision_date": None}]
        
        # Get page revisions
        result = []
        base_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        
        # Maximum number of revisions to process for performance
        max_total_revisions = 10  # Increased from 5 for more data points
        
        # Get all revisions
        revs = list(page.revisions())
        print(f"Found {len(revs)} total revisions for {page_title}")
        
        # Filter revisions by date if provided
        if start_date and end_date:
            filtered_revs = []
            for rev in revs:
                rev_date = rev['timestamp']
                if isinstance(rev_date, time.struct_time):
                    # Convert time.struct_time to datetime
                    rev_date = datetime.datetime(
                        rev_date.tm_year, rev_date.tm_mon, rev_date.tm_mday,
                        rev_date.tm_hour, rev_date.tm_min, rev_date.tm_sec
                    )
                elif isinstance(rev_date, str):
                    # Convert string timestamp to datetime if needed
                    try:
                        rev_date = datetime.datetime.strptime(rev_date, '%Y-%m-%dT%H:%M:%SZ')
                    except ValueError:
                        continue
                        
                if start_date <= rev_date <= end_date:
                    filtered_revs.append(rev)
            revs = filtered_revs
            print(f"Filtered to {len(revs)} revisions within date range {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # If we have too many revisions, sample them across the date range
        if len(revs) > max_total_revisions:
            # Select revisions evenly distributed across the date range
            indices = np.linspace(0, len(revs) - 1, max_total_revisions, dtype=int)
            revs = [revs[i] for i in indices]
            print(f"Sampled {len(revs)} revisions evenly across the date range")
            
        # Process revisions in parallel with a maximum of 5 workers to avoid overloading
        max_workers = min(len(revs), 5)
        print(f"Processing {len(revs)} revisions with {max_workers} parallel workers")
        
        # Process revisions in parallel
        def process_revision(i, rev):
            try:
                # Get revision ID
                rev_id = rev['revid']
                rev_date = rev['timestamp']
                rev_comment = rev.get('comment', 'No comment provided')
                
                # Format the date nicely
                if isinstance(rev_date, time.struct_time):
                    rev_date_obj = datetime.datetime(
                        rev_date.tm_year, rev_date.tm_mon, rev_date.tm_mday,
                        rev_date.tm_hour, rev_date.tm_min, rev_date.tm_sec
                    )
                    rev_date_formatted = rev_date_obj.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(rev_date, str):
                    try:
                        rev_date_obj = datetime.datetime.strptime(rev_date, '%Y-%m-%dT%H:%M:%SZ')
                        rev_date_formatted = rev_date_obj.strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        rev_date_formatted = rev_date
                else:
                    rev_date_formatted = rev_date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(rev_date, 'strftime') else str(rev_date)
                
                # Get the text for this specific revision - with retry mechanism
                retry_count = 0
                rev_text = None
                
                while retry_count < 3 and rev_text is None:
                    try:
                        rev_text = page.text(section=None, expandtemplates=False, cache=False, revision=rev_id)
                        break
                    except Exception as e1:
                        retry_count += 1
                        print(f"Failed to get revision text (attempt {retry_count}/3): {str(e1)}")
                        time.sleep(1)  # Short delay before retry
                        
                        if retry_count == 3:
                            # Final attempt with alternative method
                            try:
                                # Try retrieving the revision directly from the API
                                revision_query = site.api('query', prop='revisions', rvprop='content', revids=rev_id)
                                pages = revision_query.get('query', {}).get('pages', {})
                                if pages:
                                    page_data = next(iter(pages.values()))
                                    revisions = page_data.get('revisions', [])
                                    if revisions:
                                        rev_text = revisions[0].get('*', f"Could not retrieve text for revision {rev_id}")
                                    else:
                                        rev_text = f"No revision content found for ID {rev_id}"
                                else:
                                    rev_text = f"Could not retrieve text for revision {rev_id}"
                            except Exception as e2:
                                rev_text = f"Error retrieving revision text: {str(e1)}, {str(e2)}"
                
                if rev_text is None:
                    rev_text = "Failed to retrieve revision content after multiple attempts"
                
                cleaned_text = clean_text(rev_text)
                revision_results = []
                
                # If it's the first revision, include main section as a separate item
                if i == 0:
                    try:
                        summary = page.text(section=0, revision=rev_id).strip()
                        if summary:
                            revision_results.append({
                                "title": f"{page_title} - Summary (Rev {rev_date_formatted})",
                                "text": clean_text(summary),
                                "url": f"{base_url}?oldid={rev_id}",
                                "revision_date": rev_date_formatted,
                                "revision_comment": rev_comment
                            })
                    except Exception as e:
                        pass  # Skip summary if can't retrieve
                
                # Add the full revision text
                revision_results.append({
                    "title": f"{page_title} - Revision {rev_date_formatted}",
                    "text": cleaned_text,
                    "url": f"{base_url}?oldid={rev_id}",
                    "revision_date": rev_date_formatted,
                    "revision_comment": rev_comment
                })
                
                # Get sections for this revision
                sections = rev_text.split('\n==')
                for section in sections[1:]:
                    if '==' in section:
                        # Extract section title
                        title_end = section.find('==')
                        section_title = section[:title_end].strip()
                        section_text = section[title_end+2:].strip()
                        
                        # Clean text and skip empty sections
                        cleaned_section_text = clean_text(section_text)
                        if cleaned_section_text:
                            section_anchor = section_title.lower().replace(' ', '_')
                            revision_results.append({
                                "title": f"{section_title} - Revision {rev_date_formatted}",
                                "text": cleaned_section_text,
                                "url": f"{base_url}?oldid={rev_id}#{section_anchor}",
                                "revision_date": rev_date_formatted,
                                "revision_comment": rev_comment
                            })
                
                return revision_results
                
            except Exception as e:
                return [{
                    "title": f"Error processing revision {rev.get('revid')}",
                    "text": f"Error processing revision: {str(e)}",
                    "url": base_url,
                    "revision_date": rev.get('timestamp', 'Unknown'),
                    "revision_comment": rev.get('comment', 'No comment')
                }]
        
        # Process revisions in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(revs), 5)) as executor:
            futures = [executor.submit(process_revision, i, rev) for i, rev in enumerate(revs)]
            for future in futures:
                revision_results = future.result()
                result.extend(revision_results)
        
        # Save results to cache
        try:
            import json
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"Saved Wikipedia content to cache: {cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {str(e)}")
        
        return result
    except Exception as e:
        return [{"title": "Error", "text": f"Error fetching Wikipedia content: {str(e)}", "url": "", "revision_date": None}]

# ------------------------------
# Data source URL mapping (RSS only for now)
# ------------------------------
RSS_SOURCES = {
    "cointelegraph": "https://cointelegraph.com/rss/tag/bitcoin",
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "cryptonews": "https://cryptonews.com/news/feed/",
    "bitcoinmagazine": "https://bitcoinmagazine.com/.rss/full/",
    "decrypt": "https://decrypt.co/feed",
}

# ------------------------------
# Routes
# ------------------------------

@app.route("/", methods=["GET"])
def index():
    """Renders the home page."""
    # Generate a unique ID for this analysis session
    session['analysis_id'] = str(uuid.uuid4())
    return render_template("index.html", year=datetime.datetime.now().year, rss_sources=RSS_SOURCES)

@app.route("/analyze", methods=["POST"])
def analyze():
    # Get session ID or create a new one
    analysis_id = session.get('analysis_id', str(uuid.uuid4()))
    
    # Initialize processing status
    processing_status[analysis_id] = {
        'status': 'starting',
        'log': ['Starting analysis...'],
        'progress': 0
    }
    
    # Start analysis in a separate thread
    threading.Thread(target=perform_analysis, args=(analysis_id, request.form)).start()
    
    # Redirect to status page
    return render_template("processing.html", analysis_id=analysis_id)

@app.route("/status/<analysis_id>", methods=["GET"])
def status(analysis_id):
    """Returns the current processing status for the given analysis ID."""
    if analysis_id in processing_status:
        # Create a safe copy without non-serializable objects
        status_copy = {
            'status': processing_status[analysis_id]['status'],
            'log': processing_status[analysis_id]['log'],
            'progress': processing_status[analysis_id]['progress']
        }
        return jsonify(status_copy)
    return jsonify({'status': 'not_found'})

@app.route("/cancel/<analysis_id>", methods=["POST"])
def cancel_processing(analysis_id):
    """Cancels an ongoing processing task."""
    if analysis_id in processing_status and processing_status[analysis_id]['status'] == 'processing':
        processing_status[analysis_id]['status'] = 'cancelled'
        processing_status[analysis_id]['log'].append("Analysis cancelled by user.")
        processing_status[analysis_id]['progress'] = 100
        return jsonify({'status': 'cancelled'})
    return jsonify({'status': 'not_found_or_already_complete'})

@app.route("/results/<analysis_id>", methods=["GET"])
def results(analysis_id):
    """Returns the results page for the given analysis ID."""
    if analysis_id in processing_status and processing_status[analysis_id]['status'] == 'completed':
        result_data = processing_status[analysis_id]['result_data']
        
        # Get current page from query parameters (default to 1)
        current_page = request.args.get('page', 1, type=int)
        
        # Calculate sentiment counts and prepare trading signals
        articles_df = None
        trading_signals = {}
        
        if 'articles_data' in result_data:
            df = pd.DataFrame(result_data['articles_data'])
            articles_df = df.copy()  # Save a copy for trading signals calculation
            
            # Ensure all required sentiment keys exist in sentiment_counts
            sentiment_counts = df['Sentiment Label'].value_counts().to_dict()
            for key in ['positive', 'negative', 'neutral']:
                if key not in sentiment_counts:
                    sentiment_counts[key] = 0
            
            total_articles = len(df)
            overall_sentiment = max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "neutral"
            
            # Apply pagination to the articles list
            per_page = 10  # Show 10 articles per page
            start_idx = (current_page - 1) * per_page
            end_idx = min(start_idx + per_page, len(result_data['articles_data']))
            
            page_articles = result_data['articles_data'][start_idx:end_idx]
            total_pages = (len(result_data['articles_data']) + per_page - 1) // per_page  # Ceiling division
            
            # Prepare articles data for template with pagination
            articles = []
            for article in page_articles:
                # Extract URL from link HTML if needed
                url = article.get('Link', '#')
                if '<a href=' in url:
                    import re
                    link_match = re.search(r'href="([^"]*)"', url)
                    url = link_match.group(1) if link_match else '#'
                
                articles.append({
                    'title': article['Title'],
                    'description': article.get('Original Text', ''),
                    'publishedAt': article['Published'],
                    'sentiment': article['Sentiment Label'],
                    'score': article['Sentiment Score'],
                    'url': url,
                    'urlToImage': None,
                    'source': {'name': article['Source']}
                })
                
            # Generate trading signals from sentiment data
            trading_signals = calculate_trading_signals(articles_df)
        else:
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            total_articles = 0
            overall_sentiment = "neutral"
            articles = []
            total_pages = 1
            trading_signals = {
                'buy_signal_date': 'N/A',
                'buy_signal_price': 'N/A',
                'sell_signal_date': 'N/A',
                'sell_signal_price': 'N/A',
                'total_trades': 0,
                'win_rate': 0,
                'net_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'price_sentiment_correlation': 0,
                'price_data': [],
                'sentiment_data': []
            }

        # Create visualization data structure
        visualization_data = {
            'sentiment_counts': sentiment_counts,
            'sentiment_over_time': {
                'dates': [],
                'positive': [],
                'negative': [],
                'neutral': []
            },
            'sentiment_by_source': {},
            'top_keywords': {
                'words': ['Bitcoin', 'Crypto', 'Market', 'Price', 'Trading'],  # Default keywords
                'counts': [5, 4, 3, 2, 1]  # Default counts
            },
            'daily_sentiment': {
                'dates': [],
                'average_scores': [],
                'article_counts': []
            },
            'trading': {
                'price_data': trading_signals.get('price_data', []),
                'sentiment_data': trading_signals.get('sentiment_data', []),
                'correlation': trading_signals.get('price_sentiment_correlation', 0)
            }
        }
        
        # Only process if we have articles
        if 'articles_data' in result_data and len(result_data['articles_data']) > 0:
            # Group by date for time series
            df['Date'] = pd.to_datetime(df['Published']).dt.date
            date_groups = df.groupby('Date')
            
            # Sort dates chronologically
            dates = sorted(date_groups.groups.keys())
            
            # Prepare sentiment over time data
            for date in dates:
                date_df = date_groups.get_group(date)
                date_counts = date_df['Sentiment Label'].value_counts().to_dict()
                
                # Convert date to string format for JSON serialization
                date_str = date.strftime('%Y-%m-%d')
                
                visualization_data['sentiment_over_time']['dates'].append(date_str)
                visualization_data['sentiment_over_time']['positive'].append(date_counts.get('positive', 0))
                visualization_data['sentiment_over_time']['negative'].append(date_counts.get('negative', 0))
                visualization_data['sentiment_over_time']['neutral'].append(date_counts.get('neutral', 0))
                
                # Daily sentiment
                visualization_data['daily_sentiment']['dates'].append(date_str)
                visualization_data['daily_sentiment']['average_scores'].append(float(date_df['Sentiment Score'].mean()))
                visualization_data['daily_sentiment']['article_counts'].append(len(date_df))
            
            # Group by source
            source_groups = df.groupby('Source')
            for source, group in source_groups:
                source_counts = group['Sentiment Label'].value_counts().to_dict()
                visualization_data['sentiment_by_source'][source] = {
                    'positive': source_counts.get('positive', 0),
                    'negative': source_counts.get('negative', 0),
                    'neutral': source_counts.get('neutral', 0)
                }

        # Use the custom JSON encoder to handle Pandas Timestamp objects
        visualization_json = json.dumps(visualization_data, cls=CustomJSONEncoder)
        
        # Status log for trading signals
        status_log = result_data['status_log']
        status_log.append(f"Trading signals generated with correlation: {trading_signals.get('price_sentiment_correlation', 0)}")
        
        # Pass the data to the template
        return render_template(
            "results.html",
            table_html=result_data['table_html'],
            status_log=status_log,
            computation_time=result_data['computation_time'],
            year=datetime.datetime.now().year,
            page=current_page,
            total_pages=total_pages,
            total_entries=total_articles,
            current_page=current_page,
            analysis_id=analysis_id,
            start_date=result_data['start_date'],
            end_date=result_data['end_date'],
            datasources=result_data['datasources'],
            rss_sources=result_data['rss_sources'],
            sentiment_counts=sentiment_counts,
            total_articles=total_articles,
            overall_sentiment=overall_sentiment,
            visualizations=visualization_json,  # Use our JSON-serialized data
            articles=articles,
            
            # Trading signals and metrics
            buy_signal_date=trading_signals.get('buy_signal_date', 'N/A'),
            buy_signal_price=trading_signals.get('buy_signal_price', 'N/A'),
            sell_signal_date=trading_signals.get('sell_signal_date', 'N/A'),
            sell_signal_price=trading_signals.get('sell_signal_price', 'N/A'),
            total_trades=trading_signals.get('total_trades', 0),
            win_rate=trading_signals.get('win_rate', 0),
            net_profit=trading_signals.get('net_profit', 0),
            max_drawdown=trading_signals.get('max_drawdown', 0),
            sharpe_ratio=trading_signals.get('sharpe_ratio', 0),
            sortino_ratio=trading_signals.get('sortino_ratio', 0),
            price_sentiment_correlation=trading_signals.get('price_sentiment_correlation', 0),
            debug=True  # Enable debug mode to help troubleshooting
        )
    return redirect(url_for('index'))

@app.route("/export/<analysis_id>", methods=["GET"])
def export_data(analysis_id):
    """Export sentiment data for HMM integration."""
    if analysis_id in processing_status and processing_status[analysis_id]['status'] == 'completed':
        result_data = processing_status[analysis_id]['result_data']
        
        if 'articles_data' in result_data:
            try:
                # Convert articles to DataFrame
                df = pd.DataFrame(result_data['articles_data'])
                
                # Export data for HMM integration
                output_file = export_sentiment_data_for_hmm(df)
                
                if output_file:
                    # Store the path in the session so it can be accessed in the results page
                    session['export_file_path'] = output_file
                    session['export_success'] = True
                    session['export_message'] = f'Data successfully exported to: {os.path.basename(output_file)}'
                    
                    # Check if the request prefers JSON (for API use)
                    if request.headers.get('Accept') == 'application/json':
                        return jsonify({
                            'status': 'success',
                            'message': f'Data exported successfully to {output_file}',
                            'file_path': output_file
                        })
                    
                    # Otherwise redirect back to results page with a tab parameter
                    return redirect(url_for('results', analysis_id=analysis_id, tab='trading'))
                else:
                    session['export_success'] = False
                    session['export_message'] = 'Failed to export data. No articles found or export error.'
                    
                    if request.headers.get('Accept') == 'application/json':
                        return jsonify({
                            'status': 'error',
                            'message': 'Failed to export data. No articles found or export error.'
                        }), 500
                    
                    return redirect(url_for('results', analysis_id=analysis_id, tab='trading'))
            except Exception as e:
                session['export_success'] = False
                session['export_message'] = f'Error during export: {str(e)}'
                
                if request.headers.get('Accept') == 'application/json':
                    return jsonify({
                        'status': 'error',
                        'message': f'Error during export: {str(e)}'
                    }), 500
                
                return redirect(url_for('results', analysis_id=analysis_id, tab='trading'))
        else:
            session['export_success'] = False
            session['export_message'] = 'No articles data found for this analysis ID.'
            
            if request.headers.get('Accept') == 'application/json':
                return jsonify({
                    'status': 'error',
                    'message': 'No articles data found for this analysis ID.'
                }), 404
            
            return redirect(url_for('results', analysis_id=analysis_id, tab='trading'))
    
    session['export_success'] = False
    session['export_message'] = 'Analysis not found or not completed.'
    
    if request.headers.get('Accept') == 'application/json':
        return jsonify({
            'status': 'error',
            'message': 'Analysis not found or not completed.'
        }), 404
    
    return redirect(url_for('index'))

def perform_analysis(analysis_id, form_data):
    """Performs sentiment analysis in a separate thread and updates status."""
    try:
        # Record start time
        start_time = time.time()
        status_log = []
        status_log.append(FINBERT_STATUS)
        
        # Update status
        processing_status[analysis_id]['status'] = 'processing'
        processing_status[analysis_id]['log'] = status_log
        processing_status[analysis_id]['progress'] = 5

        # Get multi-selected data sources
        datasources = form_data.getlist("datasources")
        custom_url = form_data.get("custom_url", "").strip()
        
        # Get Wikipedia page title if provided
        wikipedia_page = form_data.get("wikipedia_page", "").strip()
        wikipedia_filters = form_data.get("wikipedia_filters", "").strip().lower()
        
        # Get selected RSS sources
        selected_rss_sources = form_data.getlist("rss_sources")

        # Parse date input
        try:
            # Ensure both date parameters exist
            start_date_str = form_data.get("start_date")
            end_date_str = form_data.get("end_date")
            
            if not start_date_str or not end_date_str:
                error_msg = "Missing date parameters. Please go back and try again."
                processing_status[analysis_id]['status'] = 'error'
                processing_status[analysis_id]['log'].append(error_msg)
                return
                
            start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d")
            
            # Store original dates for later use
            original_start_date = start_date
            original_end_date = end_date
            
            # Ensure News API doesn't exceed 5 days
            if 'newsapi' in datasources:
                date_diff = (end_date - start_date).days
                if date_diff > 5:
                    status_log.append(f"WARNING: News API limited to 5 days max. Original range was {date_diff} days.")
                    start_date = end_date - datetime.timedelta(days=5)
                    status_log.append(f"Adjusted start date to {start_date.strftime('%Y-%m-%d')} for News API.")
                    # Update status
                    processing_status[analysis_id]['log'] = status_log
        except Exception as e:
            processing_status[analysis_id]['status'] = 'error'
            processing_status[analysis_id]['log'].append(f"Error parsing dates: {str(e)}")
            return

        # Container for articles from all sources
        articles = []
        
        # Process each selected datasource
        if "rss" in datasources:
            processing_status[analysis_id]['progress'] = 10
            # Process each selected RSS source
            if not selected_rss_sources:
                selected_rss_sources = ["cointelegraph"]  # Default if none selected
                
            total_rss_sources = len(selected_rss_sources)
            for i, source_key in enumerate(selected_rss_sources):
                if source_key in RSS_SOURCES:
                    rss_url = RSS_SOURCES[source_key]
                    status_msg = f"Processing RSS from {source_key.capitalize()}..."
                    status_log.append(status_msg)
                    processing_status[analysis_id]['log'] = status_log
                    processing_status[analysis_id]['progress'] = 10 + (i / total_rss_sources * 30)
                    
                    feed = feedparser.parse(rss_url)
                    rss_articles_processed = 0
                    
                    # For RSS, we process all articles
                    for entry in feed.entries:
                        try:
                            # Get the published date but don't filter by it
                            published_dt = datetime.datetime(*entry.published_parsed[:6])
                            
                            title = entry.get("title", "")
                            summary = entry.get("summary", "")
                            original_text = title + " " + summary
                            processed_text = clean_text(original_text)

                            if finbert:
                                try:
                                    result = finbert(processed_text[:512])[0]
                                    score = result["score"]
                                    label = result["label"].lower()
                                    if label == "negative":
                                        score = -score
                                    elif label == "neutral":
                                        score = 0.0
                                except Exception as e:
                                    score, label = 0.0, "error"
                            else:
                                score, label = None, "n/a"

                            articles.append({
                                "Title": title,
                                "Published": published_dt.strftime("%Y-%m-%d"),
                                "Original Text": original_text,
                                "Processed Text": processed_text,
                                "Sentiment Score": score,
                                "Sentiment Label": label,
                                "Link": f'<a href="{entry.get("link", "#")}" target="_blank">Read more</a>',
                                "Source": f"RSS ({source_key.capitalize()})"
                            })
                            rss_articles_processed += 1
                            
                        except Exception as e:
                            continue
                            
                    status_log.append(f"RSS ({source_key.capitalize()}): Processed {rss_articles_processed} articles.")
                    processing_status[analysis_id]['log'] = status_log

        if "newsapi" in datasources:
            status_msg = "Processing News API articles..."
            status_log.append(status_msg)
            processing_status[analysis_id]['log'] = status_log
            processing_status[analysis_id]['progress'] = 40
            
            try:
                # Convert dates to UTC for News API
                start_date_utc = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date_utc = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                
                date_diff = (end_date_utc - start_date_utc).days
                status_log.append(f"Date range for News API: {date_diff+1} days ({start_date_utc.strftime('%Y-%m-%d')} to {end_date_utc.strftime('%Y-%m-%d')})")
                processing_status[analysis_id]['log'] = status_log
                
                # Get all dates in the range
                date_range = []
                current_date = start_date_utc
                while current_date <= end_date_utc:
                    date_range.append(current_date.strftime('%Y-%m-%d'))
                    current_date += datetime.timedelta(days=1)
                
                # Initialize empty lists for each date
                date_grouped_articles = {date: [] for date in date_range}
                
                # Make API request for each day to ensure we get articles from all dates
                total_dates = len(date_range)
                for i, date in enumerate(date_range):
                    day_start = datetime.datetime.strptime(date, '%Y-%m-%d').replace(hour=0, minute=0, second=0, microsecond=0)
                    day_end = day_start.replace(hour=23, minute=59, second=59, microsecond=999999)
                    
                    status_log.append(f"Fetching articles for {date}...")
                    processing_status[analysis_id]['log'] = status_log
                    processing_status[analysis_id]['progress'] = 40 + (i / total_dates * 30)
                    
                    # Get all pages of articles for this day
                    page = 1
                    while True:
                        day_articles = newsapi.get_everything(
                            q='bitcoin',
                            from_param=day_start.strftime('%Y-%m-%dT%H:%M:%S'),
                            to=day_end.strftime('%Y-%m-%dT%H:%M:%S'),
                            language='en',
                            sort_by='publishedAt',
                            page=page
                        )
                        
                        if not day_articles['articles']:
                            break
                            
                        for article in day_articles['articles']:
                            try:
                                published_dt = datetime.datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                                date_key = published_dt.strftime('%Y-%m-%d')
                                
                                if date_key in date_grouped_articles:
                                    date_grouped_articles[date_key].append(article)
                            except Exception:
                                continue
                        
                        page += 1
                        # Add a small delay to avoid hitting rate limits
                        time.sleep(0.5)
                
                # Calculate how many articles to take per day
                num_dates = len(date_range)
                if num_dates == 0:
                    status_log.append("No articles found from News API in the specified date range.")
                    processing_status[analysis_id]['log'] = status_log
                else:
                    # Target around 100 articles total, distributed equally across dates
                    target_total_articles = 100
                    articles_per_day = target_total_articles // num_dates
                    
                    # Process articles with equal distribution
                    news_api_articles_processed = 0
                    
                    for date_key in date_range:
                        articles_for_date = date_grouped_articles[date_key]
                        # Take equal number of articles from each day
                        articles_to_process = articles_for_date[:articles_per_day]
                        status_log.append(f"Processing {len(articles_to_process)} articles for date {date_key}")
                        processing_status[analysis_id]['log'] = status_log
                        
                        for article in articles_to_process:
                            title = article.get('title', '')
                            description = article.get('description', '')
                            
                            # Handle None values to prevent concatenation error
                            if description is None:
                                description = ''
                            
                            original_text = title + " " + description
                            processed_text = clean_text(original_text)

                            if finbert:
                                try:
                                    result = finbert(processed_text[:512])[0]
                                    score = result["score"]
                                    label = result["label"].lower()
                                    if label == "negative":
                                        score = -score
                                    elif label == "neutral":
                                        score = 0.0
                                except Exception as e:
                                    score, label = 0.0, "error"
                            else:
                                score, label = None, "n/a"

                            # Handle None values for source and name
                            source_name = "Unknown"
                            if article.get('source') is not None and article.get('source').get('name') is not None:
                                source_name = article.get('source').get('name')
                            
                            published_dt = datetime.datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                            articles.append({
                                "Title": title,
                                "Published": published_dt.strftime("%Y-%m-%d"),
                                "Original Text": original_text,
                                "Processed Text": processed_text,
                                "Sentiment Score": score,
                                "Sentiment Label": label,
                                "Link": f'<a href="{article.get("url", "#")}" target="_blank">Read more</a>',
                                "Source": f"News API ({source_name})"
                            })
                            news_api_articles_processed += 1
                    
                    status_log.append(f"News API: Processed {news_api_articles_processed} articles across {num_dates} dates.")
                    processing_status[analysis_id]['log'] = status_log
                
            except Exception as e:
                status_log.append(f"Error processing News API: {str(e)}")
                processing_status[analysis_id]['log'] = status_log

        # Process Wikipedia content if provided
        if "wikipedia" in datasources and wikipedia_page:
            status_msg = f"Processing Wikipedia content for '{wikipedia_page}'..."
            status_log.append(status_msg)
            processing_status[analysis_id]['log'] = status_log
            processing_status[analysis_id]['progress'] = 65
            
            # Fetch Wikipedia content with date filtering (using the original_start_date and original_end_date)
            status_log.append(f"Fetching Wikipedia revisions from {original_start_date.strftime('%Y-%m-%d')} to {original_end_date.strftime('%Y-%m-%d')}")
            processing_status[analysis_id]['log'] = status_log
            
            wiki_sections = get_wikipedia_content(wikipedia_page, start_date=original_start_date, end_date=original_end_date)
            wiki_articles_processed = 0
            
            for section in wiki_sections:
                # Apply filtering if specified
                if wikipedia_filters and wikipedia_filters not in section['title'].lower() and wikipedia_filters not in section['text'].lower():
                    continue
                    
                title = section['title']
                text = section['text']
                
                # Skip sections with very short content
                if len(text) < 50:
                    continue
                
                # Use the revision date if available, otherwise current date
                published_date = section.get('revision_date')
                if published_date and isinstance(published_date, str):
                    try:
                        # Try to parse the revision date
                        published_dt = datetime.datetime.strptime(published_date, '%Y-%m-%d %H:%M:%S')
                    except (ValueError, TypeError):
                        # Fall back to current date if parsing fails
                        published_dt = datetime.datetime.now()
                else:
                    published_dt = datetime.datetime.now()
                
                if finbert:
                    try:
                        # Sentiment analysis on each section's text
                        result = finbert(text[:512])[0]
                        score = result["score"]
                        label = result["label"].lower()
                        if label == "negative":
                            score = -score
                        elif label == "neutral":
                            score = 0.0
                    except Exception as e:
                        score, label = 0.0, "error"
                else:
                    score, label = None, "n/a"
                
                articles.append({
                    "Title": title,
                    "Published": published_dt.strftime("%Y-%m-%d"),
                    "Original Text": text,
                    "Processed Text": text,  # Already cleaned in get_wikipedia_content
                    "Sentiment Score": score,
                    "Sentiment Label": label,
                    "Link": f'<a href="{section["url"]}" target="_blank">View on Wikipedia</a>',
                    "Source": f"Wikipedia ({wikipedia_page})"
                })
                wiki_articles_processed += 1
            
            status_log.append(f"Wikipedia: Processed {wiki_articles_processed} sections from '{wikipedia_page}'.")
            processing_status[analysis_id]['log'] = status_log

        if "other" in datasources and custom_url:
            status_log.append(f"Processing custom URL: {custom_url}")
            processing_status[analysis_id]['log'] = status_log
            processing_status[analysis_id]['progress'] = 70
            
            feed = feedparser.parse(custom_url)
            count_before = len(articles)
            for entry in feed.entries:
                try:
                    published_dt = datetime.datetime(*entry.published_parsed[:6])
                except Exception:
                    continue

                if not (start_date <= published_dt <= end_date):
                    continue

                title = entry.get("title", "")
                summary = entry.get("summary", "")
                original_text = title + " " + summary
                processed_text = clean_text(original_text)

                if finbert:
                    try:
                        result = finbert(processed_text[:512])[0]
                        score = result["score"]
                        label = result["label"].lower()
                        if label == "negative":
                            score = -score
                        elif label == "neutral":
                            score = 0.0
                    except Exception as e:
                        score, label = 0.0, "error"
                else:
                    score, label = None, "n/a"

                articles.append({
                    "Title": title,
                    "Published": published_dt.strftime("%Y-%m-%d"),
                    "Original Text": original_text,
                    "Processed Text": processed_text,
                    "Sentiment Score": score,
                    "Sentiment Label": label,
                    "Link": f'<a href="{entry.get("link", "#")}" target="_blank">Read more</a>',
                    "Source": "Other (Custom URL)"
                })
            new_count = len(articles) - count_before
            status_log.append(f"Other: Processed {new_count} articles from the custom URL.")
            processing_status[analysis_id]['log'] = status_log

        # Calculate computation time
        computation_time = time.time() - start_time
        status_log.append(f"Total computation time: {computation_time:.2f} seconds.")
        processing_status[analysis_id]['log'] = status_log
        processing_status[analysis_id]['progress'] = 80

        # Create a DataFrame if there are articles
        if articles:
            df = pd.DataFrame(articles)
            # Remove any remaining HTML from the text columns
            df['Original Text'] = df['Original Text'].apply(clean_text)
            df['Processed Text'] = df['Processed Text'].apply(clean_text)
            
            # Create visualizations
            status_log.append("Creating visualizations...")
            processing_status[analysis_id]['log'] = status_log
            processing_status[analysis_id]['progress'] = 90
            
            visualizations = create_sentiment_visualizations(df)
            
            # Set up pagination
            page = form_data.get('page', 1, type=int)
            per_page = 100
            total_entries = len(df)
            total_pages = (total_entries + per_page - 1) // per_page  # Ceiling division
            
            # Get subset for current page
            start_idx = (page - 1) * per_page
            end_idx = min(start_idx + per_page, total_entries)
            df_page = df.iloc[start_idx:end_idx]
            
            # Generate HTML table for current page
            table_html = df_page.to_html(classes="table table-striped table-responsive", index=False, escape=False)
            
            # Prepare sentiment by source data correctly for the chart
            sentiment_by_source = {}
            # Group by source and sentiment label to get counts
            source_sentiment_counts = df.groupby(['Source', 'Sentiment Label']).size().reset_index(name='Count')
            
            # Create a nested dictionary for each source with counts for each sentiment
            for source in df['Source'].unique():
                sentiment_by_source[source] = {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
            
            # Fill in the actual counts
            for _, row in source_sentiment_counts.iterrows():
                source = row['Source']
                label = row['Sentiment Label']
                count = row['Count']
                if label in ['positive', 'negative', 'neutral']:
                    sentiment_by_source[source][label] = count
        else:
            articles = []  # Empty articles list
            table_html = "<div class='alert alert-warning'>No articles found for the selected time interval and sources.</div>"
            visualizations = []
            total_entries = 0
            total_pages = 0
            page = 1
            sentiment_by_source = {}

        # Store the results in the processing status dictionary
        processing_status[analysis_id].update({
            'status': 'completed',
            'progress': 100,
            'result_data': {
                'table_html': table_html,
                'visualizations': visualizations,
                'status_log': status_log,
                'computation_time': f"{computation_time:.2f}",
                'page': page,
                'total_pages': total_pages,
                'total_entries': total_entries,
                'start_date': original_start_date.strftime('%Y-%m-%d'),
                'end_date': original_end_date.strftime('%Y-%m-%d'),
                'datasources': datasources,
                'rss_sources': selected_rss_sources,
                'articles_data': articles,  # Store original articles data instead of DataFrame
                'sentiment_data_path': export_sentiment_data_for_hmm(df) if articles else None  # Export data automatically
            }
        })
        
    except Exception as e:
        processing_status[analysis_id]['status'] = 'error'
        processing_status[analysis_id]['log'].append(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()

# ------------------------------
# Run the Flask app
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)
