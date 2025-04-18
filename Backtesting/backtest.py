from .strategies.base import Strategy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


class Backtest:
    def __init__(self, data_filepath, strategy: Strategy, initial_cash, max_holding_period, trading_fees):
        self.data = pd.read_csv(data_filepath)
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position = 0
        self.entry_price = 0
        self.entry_index = 0
        self.signals = []
        self.equity_curve = []
        self.holding_period = 0
        self.trading_fees = trading_fees
        self.portfolio_values = []
        self.trade_logs = []
        self.max_holding_period = max_holding_period

    def run(self):
        self.reset()
        df = self.data.copy()
        df = self.strategy.set_predict_df(df)
        df = self.strategy.generate_signals()
        self.signals = self.strategy.signals

        # Convert DataFrame to NumPy array and get column indices
        df_np = df.to_numpy()
        columns = df.columns
        col_idx = {col: idx for idx, col in enumerate(columns)}

        equity = self.initial_cash

        for i in range(len(df_np)):
            row = df_np[i]
            date = row[col_idx.get('timestamp', -1)] if 'timestamp' in col_idx else None
            price = row[col_idx['close']]

            trade = 0
            pnl = 0
            drawdown = 0
            prev_position = self.position

            # Strategy decides what to do
            self.cash, self.position, self.entry_price, self.entry_index, self.holding_period, action = self.strategy.execute_trade(
                i, row, col_idx, self.cash, self.position, self.entry_price, self.entry_index, self.holding_period,
                self.trading_fees, self.max_holding_period
            )

            if action in ['buy', 'sell']:
                trade = 1

            # Calculate PnL
            if i > 0:
                prev_price = df_np[i - 1][col_idx['close']]
                price_change = (price - prev_price) / prev_price
                prev_equity = self.equity_curve[-1] if self.equity_curve else self.initial_cash
                pnl = price_change * prev_position
                equity = prev_equity + pnl
            else:
                equity = self.initial_cash
                price_change = 0

            self.equity_curve.append(equity)
            self.portfolio_values.append(equity)
            drawdown = equity - max(self.equity_curve)

            self.trade_logs.append({
                'datetime': date,
                'close': price,
                'price_change': price_change,
                'position': self.position,
                'trade': trade,
                'pnl': pnl,
                'equity': equity,
                'drawdown': drawdown
            })


    def get_performance_results(self):
        final_price = self.data['close'].iloc[-1]
        final_value = self.cash + self.position * final_price * (1 - self.trading_fees)
        total_return = (final_value - self.initial_cash) / self.initial_cash * 100
        total_trades = sum(self.trade_logs[i]['trade'] for i in range(len(self.trade_logs)))
        wins = 0
        losses = 0
        wins = 0
        buy_price = None

        for log in self.trade_logs:
            if log['trade'] == 1:  # Buy
                buy_price = log['close']
            elif log['trade'] == -1 and buy_price is not None:  # Sell
                sell_price = log['close']
                if sell_price > buy_price:
                    wins += 1
                else:
                    losses += 1
                buy_price = None  # reset for next round

        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0


        pv = np.array(self.portfolio_values)
        trade_df = pd.DataFrame(self.trade_logs)
        max_drawdown = trade_df['drawdown'].min()

        returns = np.diff(pv) / pv[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0

        results = {
            'Start Trade Date': self.data['timestamp'].iloc[0],
            'End Trade Date': self.data['timestamp'].iloc[-1],
            'Final Portfolio Value': final_value,
            'Total Return (%)': total_return,
            'Number of Trades': total_trades,  
            'Win Rate (%)': round(win_rate, 2),
            'Max Drawdown (%)': round(max_drawdown, 2),
            'Sharpe Ratio': round(sharpe_ratio, 2),
        }
                
        return results
    
    def get_trade_logs_csv(self, filename = 'backtest_trade_logs.csv'):
        df = pd.DataFrame(self.trade_logs)
        df.to_csv(filename, index=False)
        print(f"Trade logs saved to {filename}")

    def run_backtest_heatmap(self, bullish_range=None, bearish_range=None, metric='Final Portfolio Value'):
        if bullish_range is None:
            bullish_range = np.linspace(0.4, 0.7, 7)
        if bearish_range is None:
            bearish_range = np.linspace(0.3, 0.6, 7)

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
        self.cash = self.initial_cash
        self.position = 0
        self.entry_price = 0
        self.entry_index = 0
        self.signals = []
        self.equity_curve = []
        self.holding_period = 0
        self.portfolio_values = []
        self.trade_logs = []

    def set_thresholds(self, bull_thres, bear_thres):
        self.strategy.set_thresholds(bullish_threshold=bull_thres, bearish_threshold=bear_thres)