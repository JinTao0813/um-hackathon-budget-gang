from .strategies.base import Strategy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Backtest class for testing trading strategies
class Backtest:
    def __init__(self, strategy:Strategy , max_holding_period, trading_fees):  
        self.strategy = strategy # Strategy object
        self.data = None
        self.signals = [] # List to store signals
        self.equity_curve = []
        self.holding_period = 0 # Holding period for trades
        self.trading_fees = trading_fees # Trading fees per trade
        self.trade_logs = [] # List to store trade logs
        self.max_holding_period = max_holding_period # Maximum holding period for trades
        self.prev_position = 0

    def run(self): # Run the backtest
        self.reset()
        self.data = self.strategy.generate_signals() # Generate signals using the strategy
        self.signals = self.strategy.signals # Get the signals from the strategy
        for i in range(len(self.strategy.signals)): # Iterate through the signals
            today_data = self.data.iloc[i] # Get today's data
            date = today_data.get('timestamp', None)
            price = today_data['close']
            signal = self.strategy.signals[i] 

            # By default
            trade = 0
            pnl = 0
            drawdown = 0
            equity = 0
            price_change = 0
            position = 0

            if i>0: # Skip the first row
                if signal == 'buy':
                    position = 1 # Position 1 for buy
                elif signal == 'sell':
                    position = -1 # Position -1 for sell
                else:
                    position = self.prev_position # Hold the previous position
                
                trade = abs(position - self.prev_position) # Calculate trade by taking the absolute difference

                prev_data = self.data.iloc[i-1] # Get previous data

                # Trade metrics calculaions
                price_change = (price / prev_data['close']) - 1 # Using price today divide by price yesterday
                pnl = (price_change * self.prev_position) - (trade * self.trading_fees) 
                equity = sum (entry['pnl'] for entry in self.trade_logs) + pnl
                self.equity_curve.append(equity)
                drawdown = equity - max(self.equity_curve)

            self.equity_curve.append(equity) # Append the equity to the equity curve

            self.trade_logs.append({ # Append the trade log
                'datetime': date,
                'close_price': price,
                'price_change': price_change,
                'signal': signal,
                'position': position,
                'trade': trade,
                'pnl': pnl,
                'equity': equity,
                'drawdown': drawdown
            })
            self.prev_position = position


    def get_performance_results(self): # Get performance results
        trade_df = pd.DataFrame(self.trade_logs) # Convert trade logs to DataFrame

        # Calculate the performance metrics
        max_drawdown = trade_df['drawdown'].min()
        sharpe_ratio = trade_df['pnl'].mean() / trade_df['pnl'].std() * np.sqrt(365) # if 1 day = 365, if 1 hour = 24*365
        trade_per_interval = trade_df['trade'].sum() / len(trade_df)

        results = { # Store the results in a dictionary
            'Start Trade Date': self.data['timestamp'].iloc[0],
            'End Trade Date': self.data['timestamp'].iloc[-1],
            'Number of Trades': sum(trade_df['trade']),
            'Sharpe Ratio': round(sharpe_ratio, 6),
            'Max Drawdown (%)': round(max_drawdown, 6),
            'Trade per Interval': round(trade_per_interval, 6),
            'Trading Fees': self.trading_fees,
        }
                
        return results
    
    def get_trade_logs_csv(self, filename = 'backtest_trade_logs.csv'): # Get trade logs in CSV format
        df = pd.DataFrame(self.trade_logs)
        df.to_csv(filename, index=False)
        print(f"Trade logs saved to {filename}")


    def run_backtest_heatmap(self, bullish_range=None, bearish_range=None, metric='Final Portfolio Value'): # Run backtest heatmap
        if bullish_range is None:
            bullish_range = np.linspace(0.2, 0.9, 8)
        if bearish_range is None:
            bearish_range = np.linspace(0.2, 0.9, 8)

        print("bullish range: ", bullish_range)
        print("bearish range: ", bearish_range)
        results = []

        for bull in bullish_range: # Iterate through the bullish range
            row = []
            for bear in bearish_range:
                print(f"Running with bullish={bull:.2f}, bearish={bear:.2f}")
                # Set the thresholds for the strategy
                self.strategy.set_thresholds(bullish_threshold=bull, bearish_threshold=bear)
                # Create a new backtest instance with same settings but new strategy
                self.run()
                performance = self.get_performance_results()
                print(performance) 
                row.append(performance.get(metric, np.nan)) # Get the performance metric
                print(f"Result for bullish={bull:.2f}, bearish={bear:.2f}: {performance.get(metric, np.nan)}")
   
            results.append(row)

        # Plot heatmap
        print(results)
        max_value = max(max(row) for row in results)
        min_value = min(min(row) for row in results)
        heatmap_data = pd.DataFrame(results, index=np.round(bullish_range, 2), columns=np.round(bearish_range, 2))
        print('heatmap_data: ', heatmap_data)
        heatmap_data = heatmap_data.fillna(0)
        custom_cmap = LinearSegmentedColormap.from_list("custom", ["red", "yellow", "green"])
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=custom_cmap, vmin=min_value, vmax=max_value, annot_kws={"color": "black"})
        plt.title(f"{metric} Heatmap")
        plt.xlabel("Bearish Threshold")
        plt.ylabel("Bullish Threshold")
        plt.show()

    def set_data (self, filepath):
        """
        Set the data file path.
        """
        self.data = pd.read_csv(filepath)

    def set_strategy(self, strategy):
        """
        Set the strategy.
        """
        self.strategy = strategy

    def reset(self):
        """
        Reset the backtest parameters.
        """
        self.position = 0
        self.signals = []
        self.equity_curve = []
        self.holding_period = 0
        self.trade_logs = []
    
    def set_predict_filepath(self, filepath1, filepath2):
        """
        Set the prediction data file path.
        """
        self.strategy.set_predict_dataset_filepath(filepath1, filepath2)
        
    def set_best_thresholds(self, bull_thres, bear_thres): # Set the best thresholds for the strategy
        self.strategy.set_thresholds(bullish_threshold=bull_thres, bearish_threshold=bear_thres)