import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('stocks-v0')

print("env information:")
print("> shape:", env.shape)
print("> df.shape:", env.df.shape)
print("> prices.shape:", env.prices.shape)
print("> signal_features.shape:", env.signal_features.shape)
print("> max_possible_profit:", env.max_possible_profit())
#
# env.reset()
# env.render()

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]

    diff = np.insert(np.diff(prices), 0, 0)
    signal_features = env.df.loc[:, ['Close', 'Open', 'High', 'Low','Volume']].to_numpy()[start:end]
    signal_features = np.column_stack((prices, diff,signal_features))

    return prices, signal_features

class MyStockEnv(StocksEnv):
    _process_data = my_process_data

custom_env = MyStockEnv(
               df = STOCKS_GOOGL,
               window_size = 10,
               frame_bound = (10, 300))
print()
print("custom_env information:")
print("> shape:", custom_env.shape)
print("> df.shape:", custom_env.df.shape)
print("> prices.shape:", custom_env.prices.shape)
print("> signal_features.shape:", custom_env.signal_features.shape)
print("> max_possible_profit:", custom_env.max_possible_profit())
print("> trade_fee_bid_percent:",env.trade_fee_bid_percent)
print("> trade_fee_ask_percent:",env.trade_fee_ask_percent)

observation = custom_env.reset()
i = 0
while True:
    action = custom_env.action_space.sample()
    observation, reward, done, info = custom_env.step(action)
    if i==0:
        custom_env.render()
    if done:
        print(i)
        print("info:",info)
        break
    i+=1
plt.cla()
custom_env.render_all()
plt.show()

