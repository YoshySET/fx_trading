from datetime import datetime, timedelta
from typing import List, Dict

import pandas as pd
import yfinance as yf


class FXAutoTrading:
    def __init__(self, currency_pairs: List[str], initial_balance: float = 100000, stop_loss: float = 0.01,
                 take_profit: float = 0.02, limit_days: int = 7):
        self.currency_pairs = currency_pairs
        self.data: Dict[str, pd.DataFrame] = {}
        self.positions: Dict[str, Dict] = {pair: {'type': None, 'price': 0, 'size': 0, 'open_date': None} for pair in
                                           currency_pairs}
        self.trades: Dict[str, List] = {pair: [] for pair in currency_pairs}
        self.initial_balance = initial_balance
        self.balance = self.initial_balance
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.size = 1000
        self.limit_days = limit_days

    def fetch_data(self, start_date: str, end_date: str):
        """全ての通貨ペアの過去データを取得する"""
        for pair in self.currency_pairs:
            symbol = pair.replace('/', '') + '=X'
            df = yf.download(symbol, start=start_date, end=end_date)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = df.columns.str.lower()
            self.data[pair] = df
            # print(f"{pair}のデータを取得しました")

    def calculate_stochastic_rsi(self, pair: str, period: int = 14, smooth_k: int = 3, smooth_d: int = 3):
        """通貨ペアのストキャスティクスRSIを計算する"""
        df = self.data[pair]
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        stoch_rsi = (rsi - rsi.rolling(period).min()) / (rsi.rolling(period).max() - rsi.rolling(period).min())
        k = stoch_rsi.rolling(smooth_k).mean()
        d = k.rolling(smooth_d).mean()

        self.data[pair]['stoch_rsi_k'] = k
        self.data[pair]['stoch_rsi_d'] = d

    def determine_dow_trend(self, pair: str, date: pd.Timestamp):
        """ダウ理論に基づいてトレンドを判断する"""
        df = self.data[pair]
        try:
            current_price = df.loc[date, 'close']
            prev_date = df.index[df.index.get_loc(date) - 1]
            prev_price = df.loc[prev_date, 'close']

            short_ma = df['close'].rolling(window=10).mean()
            long_ma = df['close'].rolling(window=30).mean()

            if current_price > prev_price and short_ma.loc[date] > long_ma.loc[date]:
                return 1  # 上昇トレンド
            elif current_price < prev_price and short_ma.loc[date] < long_ma.loc[date]:
                return -1  # 下降トレンド
            else:
                return 0  # 明確なトレンドなし
        except KeyError:
            return 0  # データが存在しない場合

    def check_trade_condition(self, date: pd.Timestamp):
        """全ての通貨ペアのストキャスティクスRSIが同じ方向を示し、ダウ理論と一致するかチェックする"""
        directions = []
        dow_trends = []

        for pair in self.currency_pairs:
            try:
                k = self.data[pair].loc[date, 'stoch_rsi_k']
                d = self.data[pair].loc[date, 'stoch_rsi_d']

                if k > d:
                    directions.append(1)
                elif k < d:
                    directions.append(-1)
                else:
                    directions.append(0)

                dow_trends.append(self.determine_dow_trend(pair, date))
            except KeyError:
                directions.append(0)
                dow_trends.append(0)

        # デバッグ出力
        # print(f"{date}: Directions: {directions}, Dow Trends: {dow_trends}")

        # トレード条件を緩和：過半数の通貨ペアが同じ方向を示し、ダウ理論と一致する場合にトレード
        if sum(d == directions[0] for d in directions) >= len(directions) // 2 + 1 and \
                sum(t == directions[0] for t in dow_trends) >= len(dow_trends) // 2 + 1:
            return directions[0]
        else:
            return 0

    def execute_trade(self, direction: int, date: pd.Timestamp):
        """決定された方向に基づいてトレードを実行し、取引記録を保存する"""
        for pair in self.currency_pairs:
            current_price = self.data[pair].loc[date, 'close']
            if direction == 1 and self.positions[pair]['type'] != 'buy':
                self.close_position(pair, date)
                self.positions[pair] = {'type': 'buy', 'price': current_price, 'size': self.size, 'open_date': date}
                self.trades[pair].append({'type': 'buy', 'date': date, 'price': current_price})
                # print(f"{date}: {pair}を{current_price}で買い")
            elif direction == -1 and self.positions[pair]['type'] != 'sell':
                self.close_position(pair, date)
                self.positions[pair] = {'type': 'sell', 'price': current_price, 'size': self.size, 'open_date': date}
                self.trades[pair].append({'type': 'sell', 'date': date, 'price': current_price})
                # print(f"{date}: {pair}を{current_price}で売り")

    def close_position(self, pair: str, date: pd.Timestamp, forced: bool = False):
        """ポジションを決済する"""
        if self.positions[pair]['type'] is not None:
            exit_price = self.data[pair].loc[date, 'close']
            if self.positions[pair]['type'] == 'buy':
                profit = (exit_price - self.positions[pair]['price']) * self.positions[pair]['size']
            else:  # sell
                profit = (self.positions[pair]['price'] - exit_price) * self.positions[pair]['size']

            self.balance += profit
            close_type = 'forced_close' if forced else 'close'
            self.trades[pair].append({'type': close_type, 'date': date, 'price': exit_price, 'profit': profit})
            # print(f"{date}: {pair}のポジションを{exit_price}で決済, 利益: {profit:.2f}, 強制決済: {forced}")
            self.positions[pair] = {'type': None, 'price': 0, 'size': 0, 'open_date': None}

    def check_stop_loss_take_profit(self, date: pd.Timestamp):
        """損切りと利確のチェックを行う"""
        for pair in self.currency_pairs:
            if self.positions[pair]['type'] is not None:
                current_price = self.data[pair].loc[date, 'close']
                entry_price = self.positions[pair]['price']

                if self.positions[pair]['type'] == 'buy':
                    profit_percentage = (current_price - entry_price) / entry_price
                else:  # sell
                    profit_percentage = (entry_price - current_price) / entry_price

                if profit_percentage <= -self.stop_loss or profit_percentage >= self.take_profit:
                    self.close_position(pair, date)

    def check_position_expiration(self, date: pd.Timestamp):
        """ポジションの期限切れをチェックし、強制決済を行う"""
        for pair in self.currency_pairs:
            if self.positions[pair]['type'] is not None:
                open_date = self.positions[pair]['open_date']
                if (date - open_date).days >= self.limit_days:
                    self.close_position(pair, date, forced=True)

    def calculate_performance(self):
        """トレード結果を分析し、パフォーマンス指標を計算する"""
        total_profit = self.balance - self.initial_balance
        total_trades = sum(
            len([t for t in trades if t['type'] in ['close', 'forced_close']]) for trades in self.trades.values())
        winning_trades = sum(1 for trades in self.trades.values() for trade in trades if
                             trade['type'] in ['close', 'forced_close'] and trade['profit'] > 0)
        forced_closures = sum(
            1 for trades in self.trades.values() for trade in trades if trade['type'] == 'forced_close')

        print(f"===== パフォーマンス指標 =====")
        print(f"総利益: {total_profit:.2f}")
        print(f"総取引数: {total_trades}")
        print(f"勝率: {(winning_trades / total_trades * 100) if total_trades > 0 else 0:.2f}%")
        print(f"利益率: {(total_profit / self.initial_balance * 100):.2f}%")
        print(f"強制決済回数: {forced_closures}")
        print(f"最終残高: {self.balance:.2f}")

    def backtest(self):
        """過去のデータでバックテストを実行する"""
        start_date = max(
            self.data[self.currency_pairs[0]].index[30],  # 30日移動平均のため
            self.data[self.currency_pairs[0]].index[14]  # ストキャスティクスRSIのため
        )
        for date in self.data[self.currency_pairs[0]].loc[start_date:].index:
            for pair in self.currency_pairs:
                self.calculate_stochastic_rsi(pair)

            self.check_position_expiration(date)
            self.check_stop_loss_take_profit(date)

            trade_condition = self.check_trade_condition(date)
            if trade_condition != 0:
                self.execute_trade(trade_condition, date)

        # バックテスト終了時に全てのポジションを決済
        last_date = self.data[self.currency_pairs[0]].index[-1]
        for pair in self.currency_pairs:
            self.close_position(pair, last_date)

        # パフォーマンス指標を計算して表示
        self.calculate_performance()


# 入力変数
days = 365 * 1  # 検証範囲期間
configs_list = [
    {"stop_loss": 0.01, "take_profit": 0.02, "limit_days": 12},
    {"stop_loss": 0.02, "take_profit": 0.02, "limit_days": 12},
    {"stop_loss": 0.04, "take_profit": 0.02, "limit_days": 12},
    {"stop_loss": 0.08, "take_profit": 0.02, "limit_days": 12},
    {"stop_loss": 0.16, "take_profit": 0.02, "limit_days": 12},
]
currency_pairs = ['USD/JPY', 'GBP/JPY', 'EUR/JPY', 'NZD/JPY', 'AUD/JPY']

# 過去データを取得
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

trader = FXAutoTrading(currency_pairs)
trader.fetch_data(start_date, end_date)

# 検証実行
for configs in configs_list:
    print(
        f'\n【stop_loss": {configs["stop_loss"]}, "take_profit": {configs["take_profit"]}】, "limit_days": {configs["limit_days"]}】'
    )
    trader.stop_loss = configs["stop_loss"]
    trader.take_profit = configs["take_profit"]
    trader.limit_days = configs["limit_days"]
    trader.backtest()  # バックテストを実行
