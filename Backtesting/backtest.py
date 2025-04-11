import pandas as pd
import numpy as np

class Backtest:
    def __init__(self, data_filepath, strategy, initial_cash, max_holding_period, trading_fees):
        self.data = pd.read_csv(data_filepath)
        self.strategy = strategy
        self.cash = initial_cash
        self.position = 0
        self.entry_price = 0
        self.entry_index = 0
        self.signals = []
        self.trades = []
        self.equity = 0
        self.equity_curve = []
        self.holding_period = max_holding_period
        self.trading_fees = trading_fees
        self.portfolio_values = []
        self.trade_logs = []

    def run(self):
        self.strategy.generate_signals() 
        self.signals = self.strategy.signals
        for i in range(len(self.signals)):
            current_data = self.data.iloc[i]

            date = current_data.get('date') or current_data.get('datetime') or i
            close = current_data['close']
            premium_index = current_data.get('premium_index')
            trade = None
            pnl = 0

            signal = self.signals[i] if i<len(self.signals) else None
            price = current_data['close']
            # BUY action
            if signal == 'buy' and self.cash > price and self.position == 0: # Buy if buy signal occurs && we have cash && we don't have a position
                self.position = self.cash // (price * (1 + self.trading_fees)) # Calculate the number of shares we can buy
                cost = self.position * price * (1 + self.trading_fees)
                self.cash -= cost
                self.entry_price = price
                self.entry_index = i
                trade = 'buy'
                self.trades.append((trade, price, i))

            # SELL action
            elif self.signals[i] == 'sell' and self.position > 0: # Sell if sell signal occurs && we have a position
                if self.entry_index is not None and i - self.entry_index > self.holding_period:
                    sell_order = self.position * price * (1 - self.trading_fees)
                    self.cash += sell_order
                    trade = 'sell'
                    self.trades.append((trade, price, i))
                    self.position = 0
                    self.entry_price = 0
                    self.entry_index = None
       
            equity = self.cash + self.position * price
            self.equity_curve.append(equity)
            self.portfolio_values.append(equity)
            self.equity

             # Z-score and price change (can be strategy-specific; placeholder here)
            if i > 0:
                price_change = close - self.data['close'][i - 1]
            else:
                price_change = 0
            z_score = (price_change - np.mean(self.data['close'].diff().dropna())) / np.std(self.data['close'].diff().dropna())

            # Drawdown
            peak = max(self.equity_curve)
            drawdown = equity - peak

            # Append all fields to rows
            self.trade_logs.append({
                'datetime': date,
                'close': close,
                'premium_index': premium_index,
                'price_change': price_change,
                'z_score': z_score,
                'position': self.position,
                'trade': trade,
                'pnl': pnl,
                'equity': equity,
                'drawdown': drawdown
            })

    def get_performance_results(self):
        final_value = self.cash + self.position * self.data['close'].iloc[-1]
        total_return = (final_value - self.cash) / self.cash
        total_trades = len(self.trades)
        wins = 0
        losses = 0
        for i in range(0, len(self.trades)-1, 2):
            buy_price = self.trades[i][1]
            sell_price = self.trades[i+1][1]
            if sell_price > buy_price:
                wins += 1
            else:
                losses += 1
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0


        pv = np.array(self.portfolio_values)
        peak = np.maximum.accumulate(pv)
        drawdown = (peak-pv) / peak
        max_drawdown = np.max(drawdown) * 100

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
