import pandas
import numpy as np

# 2002-06-13

KS000660 = pandas.read_csv('000660.KS.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
KS005490 = pandas.read_csv('005490.KS.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
KS005930 = pandas.read_csv('005930.KS.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
KS035420 = pandas.read_csv('035420.KS.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
KS051910 = pandas.read_csv('051910.KS.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})
SnP = pandas.read_csv('SnP.csv', usecols=['Date', 'Close', 'Volume'], dtype={'Close': np.float64})

KS000660 = KS000660.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
KS005490 = KS005490.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
KS005930 = KS005930.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
KS035420 = KS035420.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
KS051910 = KS051910.fillna(0).astype(dtype={'Close': np.int32}, copy=False)
SnP = SnP.fillna(0).astype(dtype={'Close': np.int32}, copy=False)

stock = {'하이닉스': KS000660, '포스코': KS005490, '삼성전자': KS005930, '네이버': KS035420, '엘지화학': KS051910, 'SnP': SnP}


class SimulEnv:
    def __init__(self, name, mode):
        if mode == 'train':
            self.index = 0
        elif mode == 'test':
            self.index = 2000
        self.name = stock[name]
        self.action_space = ['buy', 'sell', 'hold']
        self.action_size = len(self.action_space)
        self.state_size = len(self.get_state())
        self.buy_price = []
        self.cash = 10000000
        self.start = True

    def get_state(self):
        price = [self.name.ix[self.index + i]['Close'] for i in range(7)]
        volume = [self.name.ix[self.index + i]['Volume'] for i in range(7)]
        america = [stock['SnP'].ix[self.index + i]['Close'] for i in range(7)]
        return price

    def get_current_price(self):
        return self.name.ix[self.index + 6]['Close']

    def asset_eval(self):
        amount = 0
        for price in self.buy_price:
            amount += price
        return amount + self.cash

    def get_reward(self, action):
        return 1

    def step(self, action):

        # 자산운용모델
        state = self.get_state()

        done = False

        reward = self.get_reward(action)

        self.index += 7

        # check end
        #
        #

        next_state = self.get_state()

        return next_state, reward, done

    def reset(self):
        self.index = 0
        self.stock = False
        self.cash = 10000000
        self.buy_price = 0

        return self.get_state()
