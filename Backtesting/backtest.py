from .strategies.base import Strategy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class Backtest:
    def __init__(self, strategy:Strategy , max_holding_period, trading_fees):
        self.strategy = strategy
        self.data = None
        self.signals = []
        self.equity_curve = []
        self.holding_period = 0
        self.trading_fees = trading_fees
        self.trade_logs = []
        self.max_holding_period = max_holding_period
        self.prev_position = 0

    def run(self):
        self.reset()
        self.data = self.strategy.generate_signals()
        self.signals = self.strategy.signals
        for i in range(len(self.strategy.signals)):
            today_data = self.data.iloc[i]
            date = today_data.get('timestamp', None)
            price = today_data['close']
            signal = self.strategy.signals[i] 

            trade = 0
            pnl = 0
            drawdown = 0
            equity = 0
            price_change = 0
            position = 0

            if i>0:
                if signal == 'buy':
                    position = 1
                elif signal == 'sell':
                    position = -1
                else:
                    position = self.prev_position
                
                trade = abs(position - self.prev_position)

                prev_data = self.data.iloc[i-1]

                price_change = (price / prev_data['close']) - 1
                pnl = (price_change * self.prev_position) - (trade * self.trading_fees)
                equity = sum (entry['pnl'] for entry in self.trade_logs) + pnl
                self.equity_curve.append(equity)
                drawdown = equity - max(self.equity_curve)

            self.equity_curve.append(equity)

            self.trade_logs.append({
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


    def get_performance_results(self):
        trade_df = pd.DataFrame(self.trade_logs)


        max_drawdown = trade_df['drawdown'].min()
        sharpe_ratio = trade_df['pnl'].mean() / trade_df['pnl'].std() * np.sqrt(365) # if 1 day = 365, if 1 hour = 24*365
        trade_per_interval = trade_df['trade'].sum() / len(trade_df)

        results = {
            'Start Trade Date': self.data['timestamp'].iloc[0],
            'End Trade Date': self.data['timestamp'].iloc[-1],
            'Number of Trades': sum(trade_df['trade']),
            'Sharpe Ratio': round(sharpe_ratio, 6),
            'Max Drawdown (%)': round(max_drawdown, 6),
            'Trade per Interval': round(trade_per_interval, 6),
            'Trading Fees': self.trading_fees,
        }
                
        return results
    
    def get_trade_logs_csv(self, filename = 'backtest_trade_logs.csv'):
        df = pd.DataFrame(self.trade_logs)
        df.to_csv(filename, index=False)
        print(f"Trade logs saved to {filename}")

    # def plot_graph(self):
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(self.trade_logs['timestamp'], self.trade_logs['close'], label='Price', color='blue')
    #     plt.plot(self.trade_logs['timestamp'], self.trade_logs[], label='Equity Curve', color='orange')

    #     plt.title('Backtest Results')
    #     plt.xlabel('Date')
    #     plt.ylabel('Price / Equity')
    #     plt.legend()
    #     plt.show()

    def run_backtest_heatmap(self, bullish_range=None, bearish_range=None, metric='Final Portfolio Value'):
        # if bullish_range is None:
        #     bullish_range = np.linspace(0.4, 0.7, 7)
        # if bearish_range is None:
        #     bearish_range = np.linspace(0.3, 0.6, 7)

        if bullish_range is None:
            bullish_range = np.linspace(0.4, 0.5, 2)
        if bearish_range is None:
            bearish_range = np.linspace(0.3, 0.4, 2)

        print("bullish range: ", bullish_range)
        print("bearish range: ", bearish_range)
        results = []

        for bull in bullish_range:
            row = []
            for bear in bearish_range:
                # try:
                print(f"Running with bullish={bull:.2f}, bearish={bear:.2f}")
                # Set the thresholds for the strategy
                self.strategy.set_thresholds(bullish_threshold=bull, bearish_threshold=bear)
                # Create a new backtest instance with same settings but new strategy
                self.run()
                performance = self.get_performance_results()
                print(performance)
                row.append(performance.get(metric, np.nan))
                print(f"Result for bullish={bull:.2f}, bearish={bear:.2f}: {performance.get(metric, np.nan)}")
                # except Exception as e:
                #     print(f"Failed for bull={bull:.2f}, bear={bear:.2f}: {e}")
                #     row.append(np.nan)

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
        
    def set_best_thresholds(self, bull_thres, bear_thres):
        self.strategy.set_thresholds(bullish_threshold=bull_thres, bearish_threshold=bear_thres)