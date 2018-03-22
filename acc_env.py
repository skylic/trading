import pandas as pd
import numpy as np

# 2004-01-05

KS000660 = pd.read_csv('000660.KS.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
KS005490 = pd.read_csv('005490.KS.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
KS005930 = pd.read_csv('005930.KS.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
KS035420 = pd.read_csv('035420.KS.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
KS051910 = pd.read_csv('051910.KS.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
SnP = pd.read_csv('SnP.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})

KS000660 = KS000660.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
KS005490 = KS005490.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
KS005930 = KS005930.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
KS035420 = KS035420.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
KS051910 = KS051910.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
SnP = SnP.fillna(0).astype(dtype={'Close': np.int32}, copy=False)

stock = {'하이닉스': KS000660, '포스코': KS005490, '삼성전자': KS005930, '네이버': KS035420, '엘지화학': KS051910, 'SnP': SnP}


class AccuracyEnv:
    def __init__(self, name, today='2004-01-05'):
        self.name = stock[name]
        self.index = self.name.loc[self.name['Date'] == today].index[0]
        self.init_today = today
        self.action_space = ['buy', 'sell']
        self.action_size = len(self.action_space)
        self.state_size = len(self.get_state())
        self.buy_price = []
        self.cash = 10000000

    def get_state(self):
        price = [self.name.loc[self.index + i - 7]['Close'] for i in range(7)]
        volume = [self.name.loc[self.index + i - 7]['Volume'] for i in range(7)]
        america = [stock['SnP'].loc[self.index + i - 7]['Close'] for i in range(7)]
        return price

    def get_price(self):
        return self.name.loc[self.index - 1]['Close']

    def asset_eval(self):
        asset = 0
        asset += len(self.buy_price) * self.get_price()
        asset += self.cash
        return asset

    def get_reward(self, action):
        profit_rate = 0
        # 수익금액‚
        num_of_stock = len(self.buy_price)
        if action == 1 and num_of_stock != 0:
            amount = 0
            for i in range(num_of_stock):
                price = self.buy_price.pop()
                amount += price
                self.cash += self.get_price()

            profit = self.get_price() * num_of_stock - amount

            # 수익률로 변환
            # 수익률(%) = 수익금액 / 과거 매수 금액 * 100
            profit_rate = profit / amount * 100

        elif action == 0:
            self.buy_price.append(self.get_price())
            self.cash -= self.get_price()

        else:
            pass

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
        self.index += 1

        # check end
        if self.index > 4000:
            #pd.to_datetime('2010-01-01') < -- should be changed:
            done = True
        else:
            done = False

        next_state = self.get_state()

        return next_state, reward, done

    def reset(self):
        self.buy_price = []
        self.cash = 10000000
        self.index = self.name.loc[self.name['Date'] == self.init_today].index[0]

        return self.get_state()
