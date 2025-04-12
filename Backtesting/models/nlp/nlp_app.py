from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
import pandas as pd
from scipy import stats
import datetime
import time
import os
import re
import uuid
import traceback
import threading
import json

# Import from our new models module
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Backtesting.models.nlp_model import finbert, FINBERT_STATUS, clean_text, analyze_sentiment, calculate_trading_signals, generate_default_price_data, NUMPY_AVAILABLE

# Explicitly check for NumPy before importing
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    print("WARNING: NumPy is not available. Installing it now...")
    import subprocess
    try:
        subprocess.check_call(["pip", "install", "numpy>=1.20.0"])
        import numpy as np
        NUMPY_AVAILABLE = True
        print("NumPy installed successfully!")
    except:
        print("ERROR: Failed to install NumPy. Some features may not work correctly.")
        # Create a mock numpy module with minimal functionality for essential operations
        class MockNumpy:
            def __init__(self):
                self.random = MockRandom()
            
            def array(self, data):
                return data
                
        class MockRandom:
            def uniform(self, low, high, size=None):
                if size is None:
                    return (high - low) * 0.5 + low
                return [(high - low) * 0.5 + low for _ in range(size)]
                
            def normal(self, loc=0.0, scale=1.0, size=None):
                if size is None:
                    return loc
                return [loc for _ in range(size)]
                
            def randint(self, low, high, size=None):
                if size is None:
                    return low
                return [low for _ in range(size)]
                
        np = MockNumpy()
        NUMPY_AVAILABLE = False

import mwclient
import yfinance as yf
import plotly.graph_objects as go
import feedparser  # Added missing import for RSS feed parsing

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
# Wikipedia Functions
# ------------------------------
def get_wikipedia_content(page_title, start_date=None, end_date=None):
    """
    Fetch content from Wikipedia, optionally filtering by revision date.
    
    Parameters:
    -----------
    page_title : str
        Title of the Wikipedia page
    start_date : datetime, optional
        Start date for revision filtering
    end_date : datetime, optional
        End date for revision filtering
        
    Returns:
    --------
    list of dict
        List of dictionaries containing section title, text, and URL
    """
    try:
        # Initialize MediaWiki client with English Wikipedia
        site = mwclient.Site('en.wikipedia.org')
        
        # Get the page
        page = site.pages[page_title]
        
        if not page.exists:
            return [{"title": "Error", "text": f"Wikipedia page '{page_title}' not found", "url": "#"}]
        
        # Get current text
        current_text = page.text()
        
        # Basic structure for the result
        sections = []
        
        # Get the URL for the page
        base_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
        
        # Extract sections from the text
        import re
        section_pattern = r'==\s*(.*?)\s*==\n(.*?)(?=\n==|\Z)'
        matches = re.findall(section_pattern, current_text, re.DOTALL)
        
        # If no matches, use the whole text as one section
        if not matches:
            cleaned_text = clean_text(current_text)
            sections.append({
                "title": page_title,
                "text": cleaned_text,
                "url": base_url,
                "revision_date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        else:
            # Process each section
            for title, content in matches:
                # Clean the title and content
                cleaned_title = title.strip()
                cleaned_text = clean_text(content)
                
                # Only add sections with sufficient content
                if len(cleaned_text) > 100:
                    sections.append({
                        "title": cleaned_title,
                        "text": cleaned_text,
                        "url": f"{base_url}#{cleaned_title.replace(' ', '_')}",
                        "revision_date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        # If date filtering is requested, get revision history
        if start_date and end_date:
            # Get revisions within the date range
            revisions = []
            for rev in page.revisions(start=start_date, end=end_date):
                revisions.append(rev)
            
            # If we have revisions, add a timestamp to the sections
            if revisions:
                # Use the most recent revision date within range
                latest_rev = revisions[0]
                rev_timestamp = datetime.datetime.strptime(
                    latest_rev['timestamp'], '%Y-%m-%dT%H:%M:%SZ'
                ).strftime('%Y-%m-%d %H:%M:%S')
                
                # Update all sections with this revision date
                for section in sections:
                    section['revision_date'] = rev_timestamp
        
        return sections
        
    except Exception as e:
        # Return error information
        return [{"title": "Error", "text": f"Error fetching Wikipedia content: {str(e)}", "url": "#"}]

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
# Twitter Dataset Functions
# ------------------------------
def get_twitter_dataset_info():
    """
    Get information about the Twitter dataset including date range and number of tweets.
    
    Returns:
    --------
    dict
        Dictionary with dataset information
    """
    try:
        twitter_dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'datasets', 
            'Bitcoin_tweets_dataset_2.csv'
        )
        
        # Check if file exists
        if not os.path.exists(twitter_dataset_path):
            return {
                'success': False,
                'message': 'Twitter dataset file not found'
            }
        
        # Load the dataset with optimized parameters
        twitter_df = pd.read_csv(twitter_dataset_path, 
                                parse_dates=['date'],
                                usecols=['date', 'text', 'hashtags', 'is_retweet'],
                                encoding='utf-8',
                                engine='python')
        
        # Get date range and count
        if 'date' in twitter_df.columns and not twitter_df.empty:
            min_date = twitter_df['date'].min()
            max_date = twitter_df['date'].max()
            
            # Format date range as string
            date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
            
            return {
                'success': True,
                'date_range': date_range,
                'tweet_count': len(twitter_df),
                'min_date': min_date.strftime('%Y-%m-%d'),
                'max_date': max_date.strftime('%Y-%m-%d')
            }
        else:
            return {
                'success': False,
                'message': 'Could not determine date range from dataset'
            }
    except Exception as e:
        print(f"Error loading Twitter dataset: {str(e)}")
        return {
            'success': False,
            'message': f'Error: {str(e)}'
        }

def process_twitter_data(start_date, end_date, max_tweets=None):
    """
    Process Twitter data from the CSV dataset.
    
    Parameters:
    -----------
    start_date : datetime
        Start date for filtering tweets
    end_date : datetime
        End date for filtering tweets
    max_tweets : int, optional
        Maximum number of tweets to process. If None, processes all tweets.
    
    Returns:
    --------
    list
        List of processed tweets with sentiment analysis
    """
    try:
        # Record start time for calculating estimates
        process_start_time = time.time()
        
        # Construct path to Twitter dataset
        twitter_dataset_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'datasets', 
            'Bitcoin_tweets_dataset_2.csv'
        )
        
        # Check if file exists
        if not os.path.exists(twitter_dataset_path):
            print(f"Twitter dataset not found at: {twitter_dataset_path}")
            return []
        
        # Load the dataset
        twitter_df = pd.read_csv(twitter_dataset_path, 
                                encoding='utf-8',
                                engine='python')
        
        print(f"Loaded Twitter dataset with {len(twitter_df)} total tweets")
        
        # Ensure 'date' column is datetime type
        if 'date' in twitter_df.columns:
            try:
                # Convert to datetime, handling various formats
                twitter_df['date'] = pd.to_datetime(twitter_df['date'], errors='coerce')
                
                # Drop rows with invalid dates
                initial_count = len(twitter_df)
                twitter_df = twitter_df.dropna(subset=['date'])
                dropped_count = initial_count - len(twitter_df)
                print(f"Converted date column to datetime format (dropped {dropped_count} rows with invalid dates)")
                print(f"Date column type: {twitter_df['date'].dtype}")
                
                if len(twitter_df) > 0:
                    print(f"Date range in dataset: {twitter_df['date'].min()} to {twitter_df['date'].max()}")
                    print(f"Filtering for date range: {start_date} to {end_date}")
                else:
                    print("No valid dates found in dataset after conversion")
                    return []
            except Exception as e:
                print(f"Error converting date column: {str(e)}")
                return []
        else:
            print("No 'date' column found in Twitter dataset")
            return []
        
        # Filter by date range
        filtered_df = twitter_df[(twitter_df['date'] >= start_date) & 
                                 (twitter_df['date'] <= end_date)]
        
        total_tweets_in_range = len(filtered_df)
        print(f"Found {total_tweets_in_range} tweets between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
        
        # Group tweets by day to apply the per-day limit
        filtered_df['date_only'] = filtered_df['date'].dt.date
        date_groups = filtered_df.groupby('date_only')
        print(f"Found tweets from {len(date_groups)} different days")
        
        # Limit to max 500 tweets per day
        tweets_per_day_limit = 500
        selected_tweets = []
        
        for date, group in date_groups:
            day_tweets = group.copy()
            tweets_count = len(day_tweets)
            
            if tweets_count > tweets_per_day_limit:
                print(f"Limiting tweets from {date} to {tweets_per_day_limit} (from {tweets_count})")
                day_tweets = day_tweets.sample(tweets_per_day_limit, random_state=42)
            else:
                print(f"Processing all {tweets_count} tweets from {date}")
                
            selected_tweets.append(day_tweets)
        
        # Combine all selected tweets
        if selected_tweets:
            filtered_df = pd.concat(selected_tweets)
        else:
            filtered_df = pd.DataFrame(columns=twitter_df.columns)
        
        total_tweets = len(filtered_df)
        print(f"Processing a total of {total_tweets} tweets after applying daily limits")
        
        # Apply optional tweet limit (now an overall limit after daily limit)
        if max_tweets is not None and total_tweets > max_tweets:
            # Sample randomly to get diverse tweets
            filtered_df = filtered_df.sample(max_tweets, random_state=42)
            total_tweets = len(filtered_df)
            print(f"Further limiting to {total_tweets} tweets for analysis (randomly sampled)")
        
        # Calculate batch size for processing feedback
        batch_size = max(1, min(500, total_tweets // 10))  # Show progress more frequently (every 10%)
        
        # Process tweets and perform sentiment analysis
        processed_tweets = []
        tweets_per_second = 0
        
        # Print initial progress message in format for UI parsing
        print(f"Processing Twitter data: 0% complete (0/{total_tweets} tweets)")
        print(f"Processing rate: 0.0 tweets/second")
        print(f"Estimated time remaining: Calculating...")
        
        for i, (_, tweet) in enumerate(filtered_df.iterrows()):
            # Calculate and print progress at regular intervals
            if i % batch_size == 0 or i == total_tweets - 1:
                progress_percent = int((i / total_tweets) * 100) if total_tweets > 0 else 100
                
                # Calculate processing rate and estimated time remaining
                elapsed_time = time.time() - process_start_time
                if i > 0 and elapsed_time > 0:
                    tweets_per_second = i / elapsed_time
                    remaining_tweets = total_tweets - i
                    estimated_seconds = remaining_tweets / tweets_per_second if tweets_per_second > 0 else 0
                    
                    # Format estimated time remaining
                    if estimated_seconds < 60:
                        time_remaining = f"{estimated_seconds:.1f} seconds"
                    elif estimated_seconds < 3600:
                        time_remaining = f"{estimated_seconds/60:.1f} minutes"
                    else:
                        time_remaining = f"{estimated_seconds/3600:.1f} hours"
                    
                    # Print status messages with consistent format for UI parsing
                    print(f"Processing Twitter data: {progress_percent}% complete ({i+1}/{total_tweets} tweets)")
                    print(f"Processing rate: {tweets_per_second:.1f} tweets/second")
                    print(f"Estimated time remaining: {time_remaining}")
                else:
                    print(f"Processing Twitter data: {progress_percent}% complete ({i+1}/{total_tweets} tweets)")
                    print(f"Processing rate: 0.0 tweets/second")
                    print(f"Estimated time remaining: Calculating...")
            
            # Clean the tweet text
            text = clean_text(tweet['text'])
            
            # Use the imported analyze_sentiment function
            sentiment_result = analyze_sentiment(text)
            sentiment_score = sentiment_result["score"]
            sentiment_label = sentiment_result["label"]
            
            # Get hashtags
            hashtags = tweet.get('hashtags', '')
            if isinstance(hashtags, str) and hashtags.startswith('[') and hashtags.endswith(']'):
                # Parse hashtag list if stored as string
                try:
                    hashtags = hashtags.strip('[]').replace("'", "").split(', ')
                    hashtags = ', '.join(hashtags)
                except:
                    hashtags = hashtags
            
            # Check if it's a retweet
            is_retweet = tweet.get('is_retweet', False)
            if isinstance(is_retweet, str):
                is_retweet = is_retweet.lower() == 'true'
            
            # Format published date
            published_date = tweet['date'].strftime("%Y-%m-%d")
            
            # Create tweet article object
            processed_tweets.append({
                "Title": f"Tweet {'' if not is_retweet else '(Retweet)'}",
                "Published": published_date,
                "Original Text": tweet['text'],
                "Processed Text": text,
                "Sentiment Score": sentiment_score,
                "Sentiment Label": sentiment_label,
                "Link": "#",  # No direct link to tweets in dataset
                "Source": "Twitter",
                "Hashtags": hashtags
            })
        
        # Print final status update
        print(f"Processing Twitter data: 100% complete ({total_tweets}/{total_tweets} tweets)")
        print(f"Processing complete! Analyzed {total_tweets} tweets in {time.time() - process_start_time:.1f} seconds")
        
        return processed_tweets
    
    except Exception as e:
        print(f"Error processing Twitter data: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def override_print(message, **kwargs):
    """
    Safe override for print function that avoids recursion.
    """
    import builtins
    import sys
    
    # Get original print function
    original_print = builtins.__dict__['print']
    
    # Add to log directly without calling any print function
    if 'status_log' in globals():
        status_log.append(message)
    
    # Update processing status directly if we're inside analysis
    if 'analysis_id' in globals() and analysis_id in processing_status:
        processing_status[analysis_id]['log'] = status_log.copy() if 'status_log' in globals() else []
    
    # Use sys.stdout directly to avoid any print recursion
    sys.stdout.write(str(message) + "\n")
    sys.stdout.flush()

def process_twitter_api_data(api_key, api_secret, query, start_date, end_date, max_tweets=100):
    """
    Process Twitter data from the Twitter API.
    
    Parameters:
    -----------
    api_key : str
        Twitter API key
    api_secret : str
        Twitter API secret
    query : str
        Search query (e.g., "bitcoin")
    start_date : datetime
        Start date for filtering tweets
    end_date : datetime
        End date for filtering tweets
    max_tweets : int, optional
        Maximum number of tweets to process
    
    Returns:
    --------
    list
        List of processed tweets with sentiment analysis
    """
    try:
        # Import Twitter API client
        try:
            import tweepy
        except ImportError:
            print("Tweepy library not found. Installing now...")
            import subprocess
            try:
                subprocess.check_call(["pip", "install", "tweepy>=4.0.0"])
                import tweepy
                print("Tweepy installed successfully!")
            except:
                print("Failed to install Tweepy. Twitter API functionality will not work.")
                return []
        
        # Authenticate with Twitter API
        auth = tweepy.OAuth2BearerHandler(api_key)
        api = tweepy.API(auth)
        
        # Create a client
        client = tweepy.Client(bearer_token=api_key)
        
        print(f"Searching for tweets with query: {query}")
        
        # Format dates for API query
        start_time = start_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Search for tweets
        tweets = []
        for tweet in tweepy.Paginator(
            client.search_recent_tweets,
            query=query,
            start_time=start_time,
            end_time=end_time,
            max_results=100,  # Max allowed per request
            limit=max_tweets // 100 + 1  # Get enough pages to reach max_tweets
        ).flatten(limit=max_tweets):
            tweets.append(tweet)
        
        print(f"Found {len(tweets)} tweets from Twitter API")
        
        # Process tweets and perform sentiment analysis
        processed_tweets = []
        
        for tweet in tweets:
            # Clean the tweet text
            text = clean_text(tweet.text)
            
            # Use the imported analyze_sentiment function
            sentiment_result = analyze_sentiment(text)
            sentiment_score = sentiment_result["score"]
            sentiment_label = sentiment_result["label"]
            
            # Extract hashtags if available
            hashtags = []
            try:
                if hasattr(tweet, 'entities') and tweet.entities and 'hashtags' in tweet.entities:
                    hashtags = [tag['tag'] for tag in tweet.entities['hashtags']]
            except:
                pass
            
            # Check if it's a retweet
            is_retweet = False
            try:
                is_retweet = hasattr(tweet, 'referenced_tweets') and tweet.referenced_tweets and any(
                    ref.type == 'retweeted' for ref in tweet.referenced_tweets
                )
            except:
                pass
            
            # Get created date
            created_at = datetime.datetime.now()
            try:
                if hasattr(tweet, 'created_at'):
                    created_at = tweet.created_at
            except:
                pass
            
            # Format published date
            published_date = created_at.strftime("%Y-%m-%d")
            
            # Create tweet article object
            processed_tweets.append({
                "Title": f"Tweet {'' if not is_retweet else '(Retweet)'}",
                "Published": published_date,
                "Original Text": tweet.text,
                "Processed Text": text,
                "Sentiment Score": sentiment_score,
                "Sentiment Label": sentiment_label,
                "Link": f"https://twitter.com/twitter/status/{tweet.id}" if hasattr(tweet, 'id') else "#",
                "Source": "Twitter API",
                "Hashtags": ', '.join(hashtags) if hashtags else ""
            })
        
        return processed_tweets
    
    except Exception as e:
        print(f"Error processing Twitter API data: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def process_uploaded_twitter_data(csv_data, start_date, end_date, max_tweets=500):
    """
    Process Twitter data from an uploaded CSV file.
    
    Parameters:
    -----------
    csv_data : str
        CSV data as string from uploaded file
    start_date : datetime
        Start date for filtering tweets
    end_date : datetime
        End date for filtering tweets
    max_tweets : int, optional
        Maximum number of tweets to process
    
    Returns:
    --------
    list
        List of processed tweets with sentiment analysis
    """
    try:
        # Parse CSV data from string
        import io
        csv_io = io.StringIO(csv_data)
        
        # Load the CSV data into a DataFrame
        twitter_df = pd.read_csv(csv_io, parse_dates=['date'], encoding='utf-8', engine='python')
        
        # Filter by date range
        filtered_df = twitter_df[(twitter_df['date'] >= start_date) & 
                                 (twitter_df['date'] <= end_date)]
        
        print(f"Found {len(filtered_df)} tweets in uploaded CSV between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
        
        # Limit to max_tweets
        if len(filtered_df) > max_tweets:
            # Sample randomly to get diverse tweets
            filtered_df = filtered_df.sample(max_tweets, random_state=42)
            print(f"Limiting to {max_tweets} tweets from uploaded CSV for analysis")
        
        # Process tweets and perform sentiment analysis
        processed_tweets = []
        
        for _, tweet in filtered_df.iterrows():
            # Make sure we have the 'text' column
            if 'text' not in tweet:
                continue
                
            # Clean the tweet text
            text = clean_text(tweet['text'])
            
            # Use the imported analyze_sentiment function
            sentiment_result = analyze_sentiment(text)
            sentiment_score = sentiment_result["score"]
            sentiment_label = sentiment_result["label"]
            
            # Get hashtags if available
            hashtags = tweet.get('hashtags', '')
            if isinstance(hashtags, str) and hashtags.startswith('[') and hashtags.endswith(']'):
                # Parse hashtag list if stored as string
                try:
                    hashtags = hashtags.strip('[]').replace("'", "").split(', ')
                    hashtags = ', '.join(hashtags)
                except:
                    hashtags = hashtags
            
            # Check if it's a retweet if available
            is_retweet = tweet.get('is_retweet', False)
            if isinstance(is_retweet, str):
                is_retweet = is_retweet.lower() == 'true'
            
            # Format published date
            published_date = tweet['date'].strftime("%Y-%m-%d")
            
            # Create tweet article object
            processed_tweets.append({
                "Title": f"Tweet {'' if not is_retweet else '(Retweet)'}",
                "Published": published_date,
                "Original Text": tweet['text'],
                "Processed Text": text,
                "Sentiment Score": sentiment_score,
                "Sentiment Label": sentiment_label,
                "Link": "#",  # No direct link to tweets in uploaded CSV
                "Source": "Twitter (Uploaded CSV)",
                "Hashtags": hashtags
            })
        
        return processed_tweets
    
    except Exception as e:
        print(f"Error processing uploaded Twitter data: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

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
    try:
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
    except Exception as e:
        print(f"Error creating time visualization: {str(e)}")
        # Create an empty placeholder
        fig_empty = go.Figure()
        fig_empty.add_annotation(text="No time data available", showarrow=False)
        visualization_html.append(fig_empty.to_html(full_html=False, include_plotlyjs=False))
    
    # 2. Source-wise Sentiment Comparison - enhanced
    try:
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
    except Exception as e:
        print(f"Error creating source comparison: {str(e)}")
        fig_empty = go.Figure()
        fig_empty.add_annotation(text="No source data available", showarrow=False)
        visualization_html.append(fig_empty.to_html(full_html=False, include_plotlyjs=False))
    
    # 3. Sentiment Distribution
    try:
        # Create sentiment distribution histogram
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=df['Sentiment Score'],
            marker_color='rgba(58, 71, 80, 0.6)',
            nbinsx=20,
            name='Sentiment Distribution'
        ))
        
        # Check for enough unique values before creating KDE
        unique_values = df['Sentiment Score'].nunique()
        print(f"Found {unique_values} unique sentiment values")
        
        if unique_values >= 2:
            # Create data for KDE curve
            hist_data, bin_edges = np.histogram(
                df['Sentiment Score'], 
                bins=20, 
                range=(df['Sentiment Score'].min(), df['Sentiment Score'].max()),
                density=True
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            try:
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
            except Exception as kde_error:
                print(f"KDE calculation error: {str(kde_error)}")
                # Continue without KDE curve
        else:
            print(f"Not enough unique sentiment values for KDE")
        
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
    except Exception as e:
        print(f"Error creating distribution visualization: {str(e)}")
        fig_empty = go.Figure()
        fig_empty.add_annotation(text="No distribution data available", showarrow=False)
        visualization_html.append(fig_empty.to_html(full_html=False, include_plotlyjs=False))
    
    # 4. Daily Sentiment Trends
    try:
        # Calculate daily average sentiment
        df['Date'] = df['Published'].dt.date
        daily_sentiment = df.groupby('Date')['Sentiment Score'].mean().reset_index()
        daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
        daily_sentiment = daily_sentiment.sort_values('Date')
        
        # Check if we have enough data
        if len(daily_sentiment) >= 2:
            # Calculate rolling metrics
            window_size = min(7, len(daily_sentiment))
            daily_sentiment['MA7'] = daily_sentiment['Sentiment Score'].rolling(window=window_size, min_periods=1).mean()
            daily_sentiment['Volatility'] = daily_sentiment['Sentiment Score'].rolling(window=window_size, min_periods=1).std()
            
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
            
            # Only add volatility bands if we have enough data
            if daily_sentiment['Volatility'].notna().sum() > 0:
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
                    text='Sentiment Trend Analysis',
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
        else:
            print("Not enough daily sentiment data for time series visualization")
            fig_empty = go.Figure()
            fig_empty.add_annotation(text="Insufficient data for trend analysis", showarrow=False)
            visualization_html.append(fig_empty.to_html(full_html=False, include_plotlyjs=False))
    except Exception as e:
        print(f"Error creating volatility visualization: {str(e)}")
        fig_empty = go.Figure()
        fig_empty.add_annotation(text="Could not create trend analysis", showarrow=False)
        visualization_html.append(fig_empty.to_html(full_html=False, include_plotlyjs=False))
    
    return visualization_html

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
            'log': processing_status[analysis_id]['log'][-50:],  # Return only the last 50 log entries to avoid too much data
            'progress': processing_status[analysis_id]['progress'],
            'twitterComplete': processing_status[analysis_id].get('twitter_complete', False)  # Add Twitter completion status
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
            if 'Sentiment Label' in df.columns:
                sentiment_counts = df['Sentiment Label'].value_counts().to_dict()
            else:
                # Handle case when Sentiment Label column is missing
                # Check if we have a label column with different name
                if 'sentiment_label' in df.columns:
                    sentiment_counts = df['sentiment_label'].value_counts().to_dict()
                elif 'label' in df.columns:
                    sentiment_counts = df['label'].value_counts().to_dict()
                else:
                    # Create default values if no sentiment column exists
                    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            # Ensure all expected sentiment categories exist
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

@app.route("/twitter_dataset_info", methods=["GET"])
def twitter_dataset_info():
    """Returns information about the Twitter dataset."""
    info = get_twitter_dataset_info()
    return jsonify(info)

@app.route("/analyze_twitter_csv", methods=["POST"])
def analyze_twitter_csv():
    """
    Analyzes a Twitter CSV file uploaded by the user and returns basic information.
    """
    try:
        # Get CSV data from request
        data = request.json
        csv_data = data.get('csv_data', '')
        
        if not csv_data:
            return jsonify({
                'success': False,
                'message': 'No CSV data provided'
            })
        
        # Parse CSV data using pandas with more robust error handling
        import io
        csv_io = io.StringIO(csv_data)
        try:
            # First, try with small sample to detect schema
            sample_df = pd.read_csv(csv_io, nrows=5, on_bad_lines='skip', encoding='utf-8')
            csv_io.seek(0)  # Reset file pointer
            
            # Check if 'date' column exists
            if 'date' not in sample_df.columns:
                return jsonify({
                    'success': False,
                    'message': "CSV file must contain a 'date' column"
                })
            
            # Use more robust parsing options
            df = pd.read_csv(
                csv_io, 
                parse_dates=['date'],
                encoding='utf-8',
                engine='python',
                on_bad_lines='skip',
                low_memory=True,
                encoding_errors='replace',
                dtype={'text': str, 'hashtags': str}
            )
            
            # Get date range and count
            if 'date' in df.columns and not df.empty:
                min_date = df['date'].min()
                max_date = df['date'].max()
                
                # Format date range as string
                date_range = f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}"
                
                # Store dataframe in session for later use
                session['twitter_uploaded_csv'] = csv_data
                
                return jsonify({
                    'success': True,
                    'date_range': date_range,
                    'tweet_count': len(df),
                    'min_date': min_date.strftime('%Y-%m-%d'),
                    'max_date': max_date.strftime('%Y-%m-%d')
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Could not determine date range from CSV'
                })
                
        except Exception as e:
            print(f"Error parsing CSV: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Error parsing CSV: {str(e)}'
            })
            
    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        })

def perform_analysis(analysis_id, form_data):
    """Performs sentiment analysis in a separate thread and updates status."""
    try:
        # Record start time
        start_time = time.time()
        status_log = []
        status_log.append(FINBERT_STATUS)
        
        # Flag to track Twitter processing status
        twitter_processing_complete = True
        twitter_selected = "twitter" in form_data.getlist("datasources")
        
        # Function to add log messages with immediate status update
        def add_log(message):
            status_log.append(message)
            # Immediately update the processing_status log
            processing_status[analysis_id]['log'] = status_log.copy()
            # Use sys.stdout directly to avoid recursion with print
            import sys
            sys.stdout.write(str(message) + "\n")
            sys.stdout.flush()
        
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
                add_log(error_msg)
                processing_status[analysis_id]['status'] = 'error'
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
                    add_log(f"WARNING: News API limited to 5 days max. Original range was {date_diff} days.")
                    start_date = end_date - datetime.timedelta(days=5)
                    add_log(f"Adjusted start date to {start_date.strftime('%Y-%m-%d')} for News API.")
        except Exception as e:
            processing_status[analysis_id]['status'] = 'error'
            add_log(f"Error parsing dates: {str(e)}")
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
                    add_log(f"Processing RSS from {source_key.capitalize()}...")
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

                            sentiment_result = analyze_sentiment(processed_text)
                            sentiment_score = sentiment_result["score"]
                            sentiment_label = sentiment_result["label"]

                            articles.append({
                                "Title": title,
                                "Published": published_dt.strftime("%Y-%m-%d"),
                                "Original Text": original_text,
                                "Processed Text": processed_text,
                                "Sentiment Score": sentiment_score,
                                "Sentiment Label": sentiment_label,
                                "Link": f'<a href="{entry.get("link", "#")}" target="_blank">Read more</a>',
                                "Source": f"RSS ({source_key.capitalize()})"
                            })
                            rss_articles_processed += 1
                            
                        except Exception as e:
                            continue
                            
                    add_log(f"RSS ({source_key.capitalize()}): Processed {rss_articles_processed} articles.")

        if "newsapi" in datasources:
            add_log("Processing News API articles...")
            processing_status[analysis_id]['progress'] = 40
            
            try:
                # Convert dates to UTC for News API
                start_date_utc = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date_utc = end_date.replace(hour=23, minute=59, second=59, microsecond=999999)
                
                date_diff = (end_date_utc - start_date_utc).days
                add_log(f"Date range for News API: {date_diff+1} days ({start_date_utc.strftime('%Y-%m-%d')} to {end_date_utc.strftime('%Y-%m-%d')})")
                
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
                    
                    add_log(f"Fetching articles for {date}...")
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
                    add_log("No articles found from News API in the specified date range.")
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
                        add_log(f"Processing {len(articles_to_process)} articles for date {date_key}")
                        
                        for article in articles_to_process:
                            title = article.get('title', '')
                            description = article.get('description', '')
                            
                            # Handle None values to prevent concatenation error
                            if description is None:
                                description = ''
                            
                            original_text = title + " " + description
                            processed_text = clean_text(original_text)

                            sentiment_result = analyze_sentiment(processed_text)
                            sentiment_score = sentiment_result["score"]
                            sentiment_label = sentiment_result["label"]

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
                                "Sentiment Score": sentiment_score,
                                "Sentiment Label": sentiment_label,
                                "Link": f'<a href="{article.get("url", "#")}" target="_blank">Read more</a>',
                                "Source": f"News API ({source_name})"
                            })
                            news_api_articles_processed += 1
                    
                    add_log(f"News API: Processed {news_api_articles_processed} articles across {num_dates} dates.")
                
            except Exception as e:
                add_log(f"Error processing News API: {str(e)}")

        # Process Wikipedia content if provided
        if "wikipedia" in datasources and wikipedia_page:
            add_log(f"Processing Wikipedia content for '{wikipedia_page}'...")
            processing_status[analysis_id]['progress'] = 65
            
            # Fetch Wikipedia content with date filtering (using the original_start_date and original_end_date)
            add_log(f"Fetching Wikipedia revisions from {original_start_date.strftime('%Y-%m-%d')} to {original_end_date.strftime('%Y-%m-%d')}")
            
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
                
                sentiment_result = analyze_sentiment(text)
                sentiment_score = sentiment_result["score"]
                sentiment_label = sentiment_result["label"]
                
                articles.append({
                    "Title": title,
                    "Published": published_dt.strftime("%Y-%m-%d"),
                    "Original Text": text,
                    "Processed Text": text,  # Already cleaned in get_wikipedia_content
                    "Sentiment Score": sentiment_score,
                    "Sentiment Label": sentiment_label,
                    "Link": f'<a href="{section["url"]}" target="_blank">View on Wikipedia</a>',
                    "Source": f"Wikipedia ({wikipedia_page})"
                })
                wiki_articles_processed += 1
            
            add_log(f"Wikipedia: Processed {wiki_articles_processed} sections from '{wikipedia_page}'.")

        if "other" in datasources and custom_url:
            add_log(f"Processing custom URL: {custom_url}")
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

                sentiment_result = analyze_sentiment(processed_text)
                sentiment_score = sentiment_result["score"]
                sentiment_label = sentiment_result["label"]

                articles.append({
                    "Title": title,
                    "Published": published_dt.strftime("%Y-%m-%d"),
                    "Original Text": original_text,
                    "Processed Text": processed_text,
                    "Sentiment Score": sentiment_score,
                    "Sentiment Label": sentiment_label,
                    "Link": f'<a href="{entry.get("link", "#")}" target="_blank">Read more</a>',
                    "Source": "Other (Custom URL)"
                })
            new_count = len(articles) - count_before
            add_log(f"Other: Processed {new_count} articles from the custom URL.")

        if "twitter" in datasources:
            add_log("Starting Twitter data processing...")
            processing_status[analysis_id]['progress'] = 75
            
            # Set flag to indicate Twitter processing has started but not completed
            twitter_processing_complete = False
            
            # Get the Twitter source option (existing dataset or upload)
            twitter_source = form_data.get("twitter_source", "dataset")
            twitter_articles = []  # Initialize empty list to collect twitter articles
            
            if twitter_source == "dataset":
                # Process the existing Twitter dataset
                # Override the normal print function for process_twitter_data with our add_log function
                original_print = print
                def override_print(message, **kwargs):
                    """
                    Safe override for print function that avoids recursion.
                    """
                    import builtins
                    import sys
                    
                    # Get original print function
                    original_print = builtins.__dict__['print']
                    
                    # Add to log directly without calling any print function
                    if 'status_log' in globals():
                        status_log.append(message)
                    
                    # Update processing status directly if we're inside analysis
                    if 'analysis_id' in globals() and analysis_id in processing_status:
                        processing_status[analysis_id]['log'] = status_log.copy() if 'status_log' in globals() else []
                    
                    # Use sys.stdout directly to avoid any print recursion
                    sys.stdout.write(str(message) + "\n")
                    sys.stdout.flush()
                
                # Monkey patch print for this block only
                import builtins
                builtins.print = override_print
                
                try:
                    add_log("Starting Twitter dataset analysis - please wait...")
                    twitter_articles = process_twitter_data(start_date, end_date)
                    if twitter_articles:
                        articles.extend(twitter_articles)
                        add_log(f"COMPLETED: Twitter analysis processed {len(twitter_articles)} tweets from existing dataset.")
                        # Set flag indicating Twitter processing is complete
                        twitter_processing_complete = True
                    else:
                        add_log("Twitter: No tweets found in dataset for selected date range.")
                        twitter_processing_complete = True  # No tweets but processing is complete
                except Exception as tw_e:
                    add_log(f"Twitter processing error: {str(tw_e)}")
                    traceback.print_exc()
                    # Set flag to true since we're done (even with an error)
                    twitter_processing_complete = True
                finally:
                    # Restore the original print function
                    builtins.print = original_print
                    
            elif twitter_source == "upload" and 'twitter_csv_data' in form_data:
                # Process user-uploaded Twitter CSV
                uploaded_csv_data = form_data.get('twitter_csv_data')
                if uploaded_csv_data:
                    try:
                        add_log("Starting uploaded Twitter CSV analysis - please wait...")
                        uploaded_twitter_articles = process_uploaded_twitter_data(uploaded_csv_data, start_date, end_date)
                        if uploaded_twitter_articles:
                            articles.extend(uploaded_twitter_articles)
                            add_log(f"COMPLETED: Twitter analysis processed {len(uploaded_twitter_articles)} tweets from uploaded CSV.")
                        else:
                            add_log("Twitter: No tweets found in uploaded CSV for selected date range.")
                        twitter_processing_complete = True
                    except Exception as tw_e:
                        add_log(f"Twitter upload processing error: {str(tw_e)}")
                        traceback.print_exc()
                        twitter_processing_complete = True
                else:
                    add_log("Twitter: No CSV data found in upload.")
                    twitter_processing_complete = True
            elif twitter_source == "api":
                # Process Twitter data from API
                twitter_api_key = form_data.get("twitter_api_key", "").strip()
                twitter_api_secret = form_data.get("twitter_api_secret", "").strip()
                twitter_query = form_data.get("twitter_query", "bitcoin").strip()
                
                if twitter_api_key and twitter_api_secret:
                    try:
                        add_log("Starting Twitter API analysis - please wait...")
                        twitter_api_articles = process_twitter_api_data(
                            twitter_api_key, twitter_api_secret, twitter_query, start_date, end_date
                        )
                        if twitter_api_articles:
                            articles.extend(twitter_api_articles)
                            add_log(f"COMPLETED: Twitter API analysis processed {len(twitter_api_articles)} tweets.")
                        else:
                            add_log("Twitter API: No tweets found for selected date range and query.")
                        twitter_processing_complete = True
                    except Exception as tw_e:
                        add_log(f"Twitter API processing error: {str(tw_e)}")
                        traceback.print_exc()
                        twitter_processing_complete = True
                else:
                    add_log("Twitter API: Missing API key or secret.")
                    twitter_processing_complete = True
            else:
                add_log("Twitter: Using default dataset (no upload found).")
                
                # Override the normal print function for process_twitter_data
                original_print = print
                def override_print(message, **kwargs):
                    """
                    Safe override for print function that avoids recursion.
                    """
                    import builtins
                    import sys
                    
                    # Get original print function
                    original_print = builtins.__dict__['print']
                    
                    # Add to log directly without calling any print function
                    if 'status_log' in globals():
                        status_log.append(message)
                    
                    # Update processing status directly if we're inside analysis
                    if 'analysis_id' in globals() and analysis_id in processing_status:
                        processing_status[analysis_id]['log'] = status_log.copy() if 'status_log' in globals() else []
                    
                    # Use sys.stdout directly to avoid any print recursion
                    sys.stdout.write(str(message) + "\n")
                    sys.stdout.flush()
                
                # Monkey patch print for this block only
                import builtins
                builtins.print = override_print
                
                try:
                    add_log("Starting Twitter dataset analysis - please wait...")
                    twitter_articles = process_twitter_data(start_date, end_date)
                    if twitter_articles:
                        articles.extend(twitter_articles)
                        add_log(f"COMPLETED: Twitter analysis processed {len(twitter_articles)} tweets from default dataset.")
                    else:
                        add_log("Twitter: No tweets found in dataset for selected date range.")
                    twitter_processing_complete = True
                except Exception as tw_e:
                    add_log(f"Twitter processing error: {str(tw_e)}")
                    traceback.print_exc()
                    twitter_processing_complete = True
                finally:
                    # Restore the original print function
                    builtins.print = original_print
            
            # Make sure we have at least processed Twitter data before proceeding
            if "twitter" in datasources and len(twitter_articles) == 0:
                add_log("Warning: No Twitter data was successfully processed. Results may be limited.")
                
            # Explicit confirmation that Twitter processing has completed
            add_log("Twitter processing stage complete.")
                
        # Calculate computation time
        computation_time = time.time() - start_time
        add_log(f"Total computation time: {computation_time:.2f} seconds.")
        
        # Only proceed if Twitter processing is complete (if Twitter was selected)
        if twitter_selected and not twitter_processing_complete:
            add_log("ERROR: Twitter processing did not complete properly. Unable to generate results.")
            processing_status[analysis_id]['status'] = 'error'
            return
            
        processing_status[analysis_id]['progress'] = 80

        # Create a DataFrame if there are articles
        if articles:
            df = pd.DataFrame(articles)
            # Remove any remaining HTML from the text columns
            df['Original Text'] = df['Original Text'].apply(clean_text)
            df['Processed Text'] = df['Processed Text'].apply(clean_text)
            
            # Create visualizations
            add_log("Creating visualizations...")
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

        # Add a final confirmation that all processing is complete
        add_log("All data sources have been fully processed. Results are ready.")
        
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
            },
            'twitter_complete': twitter_processing_complete  # Add Twitter completion status
        })
        
    except Exception as e:
        processing_status[analysis_id]['status'] = 'error'
        processing_status[analysis_id]['log'].append(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def export_sentiment_data_for_hmm(sentiment_df, output_path=None):
    """
    Export sentiment data for analysis.
    This creates a CSV file with daily sentiment features.
    
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
    if sentiment_df is None or sentiment_df.empty:
        print("No sentiment data to export")
        return None
        
    try:
        # Ensure we have datetime
        sentiment_df['Date'] = pd.to_datetime(sentiment_df['Published'])
        
        # Group by date and calculate various sentiment metrics
        daily_data = sentiment_df.groupby(sentiment_df['Date'].dt.date).agg({
            'Sentiment Score': ['mean', 'count']
        }).reset_index()
        
        # Flatten the multi-level columns
        daily_data.columns = ['Date', 'sentiment_mean', 'article_count']
        
        # Convert to datetime for easy sorting
        daily_data['Date'] = pd.to_datetime(daily_data['Date'])
        daily_data = daily_data.sort_values('Date')
        
        # Calculate additional features
        daily_data['positive_ratio'] = sentiment_df.groupby(sentiment_df['Date'].dt.date)['Sentiment Label'].apply(
            lambda x: (x == 'positive').mean()
        ).reset_index(drop=True)
        
        daily_data['negative_ratio'] = sentiment_df.groupby(sentiment_df['Date'].dt.date)['Sentiment Label'].apply(
            lambda x: (x == 'negative').mean()
        ).reset_index(drop=True)
        
        # Add rolling window calculations if we have enough data
        if len(daily_data) >= 2:
            # Add simple momentum (day-over-day change)
            daily_data['sentiment_momentum'] = daily_data['sentiment_mean'].diff()
            
            # Calculate rolling statistics if we have enough days
            if len(daily_data) >= 3:
                window_size = min(3, len(daily_data))
                daily_data['sentiment_ma3'] = daily_data['sentiment_mean'].rolling(window=window_size, min_periods=1).mean()
                daily_data['sentiment_std'] = daily_data['sentiment_mean'].rolling(window=window_size, min_periods=1).std()
        
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
        traceback.print_exc()
        return None

# ------------------------------
# Run the Flask app
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True)