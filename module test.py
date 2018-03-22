import pandas as pd
import numpy as np

# 2004-01-05

KS000660 = pd.read_csv('000660.KS.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
KS005490 = pd.read_csv('005490.KS.csv', index_col=['Date'], usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
KS005930 = pd.read_csv('005930.KS.csv', index_col=['Date'], usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
KS035420 = pd.read_csv('035420.KS.csv', index_col=['Date'], usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
KS051910 = pd.read_csv('051910.KS.csv', index_col=['Date'], usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
SnP = pd.read_csv('SnP.csv', index_col=['Date'], usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})

KS000660 = KS000660.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
KS005490 = KS005490.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
KS005930 = KS005930.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
KS035420 = KS035420.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
KS051910 = KS051910.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
SnP = SnP.fillna(0).astype(dtype={'Close': np.int32}, copy=False)

print(KS000660.loc[KS000660['Date'] == '2004-01-05'].index[0])

stock = {'하이닉스': KS000660, '포스코': KS005490, '삼성전자': KS005930, '네이버': KS035420, '엘지화학': KS051910, 'SnP': SnP}

"""
class AccuracyEnv:
    def __init__(self, name, today):
        self.name = stock[name]
        self.today = pd.to_datetime(today)
        self.action_space = ['buy', 'sell', 'hold']
        self.action_size = len(self.action_space)
        self.state_size = len(self.get_state())
        self.buy_price = []
        self.date_range = 7
        self.cash = 10000000

    def get_state(self):
        price = [self.name.ix[self.today + pd.Timedelta(days=i - self.date_range)]['Close'] for i in range(self.date_range)]
        volume = [self.name.ix[self.today + pd.Timedelta(days=i - self.date_range)]['Volume'] for i in range(self.date_range)]
        america = [stock['SnP'].ix[self.today + pd.Timedelta(days=i - self.date_range)]['Close'] for i in range(self.date_range)]
        return price

    def get_price(self):
        return self.name.ix[self.today + pd.Timedelta(-1)]['Close']

    def asset_eval(self):
        asset = 0
        for price in self.buy_price:
            asset += price
        asset += self.cash

    def get_reward(self, action):
        profit = 0
        early_buy_price = 0
        # 수익금액
        # 현재 매도 금액 - 과거 매수 금액
        if action == 1:
            early_buy_price = self.buy_price.pop(0)
            profit = self.get_price() - early_buy_price

        elif action == 0:
            self.buy_price.append(self.get_price())
            self.cash -= self.get_price()

        # 수익률로 변환
        # 수익률(%) = 수익금액 / 과거 매수 금액 * 100
        profit_rate = profit / early_buy_price * 100

        # reward function
        # f(수익률) = reward
        if profit_rate > 0:
            reward = pow(profit_rate, 1/3)
        else:
            reward = (-1) * pow(profit_rate, 3)

        return reward

    def step(self, action):

        reward = self.get_reward(action)

        # time interval
        self.today += pd.Timedelta(days=1)

        # check end
        if self.today > pd.to_datetime('2010-01-01'):
            done = True
        else:
            done = False

        next_state = self.get_state()

        return next_state, reward, done

    def reset(self):

        return self.get_state()
        """
