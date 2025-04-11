import pandas as pd
import numpy as np

class Backtest:
    def __init__(self, data_filepath, strategy, initial_cash, max_holding_period, trading_fees):
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
        self.strategy.generate_signals() 
        self.signals = self.strategy.signals
        for i in range(len(self.signals)):
            today_data = self.data.iloc[i]
            price_change = 0
            pnl = 0
            trade = 0
            equity = 0
            drawdown = 0
            date = today_data.get('timestamp', None)


            if i > 0: # Not the first row of data
                ytd_data = self.data.iloc[i-1]
                price_change = (today_data['close'] - ytd_data['close']) / ytd_data['close']

                signal = self.signals[i] if i<len(self.signals) else None
                print(f"Signal: {signal}")
                price = today_data['close']
                # BUY action
                if signal == 'bullish' and self.cash > (price * self.trading_fees) and self.position == 0: # Buy if buy signal occurs && we have cash && we don't have a position
                    self.position = self.cash // (price * (1 + self.trading_fees)) # Calculate the number of shares we can buy
                    cost = self.position * price * (1 + self.trading_fees)
                    self.cash -= cost
                    self.entry_price = price
                    self.entry_index = i
                    trade = 1

                # SELL action
                elif self.signals[i] == 'bearish' or self.holding_period == self.max_holding_period: # Sell if sell signal occurs && we have a position
                    print(f"Sell signal at index {i}")
                    if self.entry_index is not None:
                        sell_order = self.position * price * (1 - self.trading_fees)
                        self.cash += sell_order
                        trade = 1
                        self.position = 0
                        self.entry_price = 0
                        self.entry_index = None
                        self.holding_period = 0
                else:
                    self.holding_period += 1
        
                pnl = price_change * self.trade_logs[i-1]['position'] - trade * self.trading_fees
                equity = self.trade_logs[i-1]['equity'] + pnl
                self.equity_curve.append(equity)
                self.portfolio_values.append(equity)
                drawdown = equity - max(self.equity_curve)

            print(f"Price change: {price_change}")
                
            # Append all fields to rows
            self.trade_logs.append({
                'datetime': date,
                'close': today_data['close'],
                'price_change': price_change,
                'position': self.position,
                'trade': trade,
                'pnl': pnl,
                'equity': equity,
                'drawdown': drawdown
            })

            print (f"Latest Trade logs: {self.trade_logs[-1]}")

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
