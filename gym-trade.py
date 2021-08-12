import gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL
import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import os
import gym
import numpy as np
import parl
from parl.utils import logger

# env = gym.make('stocks-v0')
#
# print("env information:")
# print("> shape:", env.shape)
# print("> df.shape:", env.df.shape)
# print("> prices.shape:", env.prices.shape)
# print("> signal_features.shape:", env.signal_features.shape)
# print("> max_possible_profit:", env.max_possible_profit())
#
# env.reset()
# env.render()

def my_process_data(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Close'].to_numpy()[start:end]

    diff = np.insert(np.diff(prices), 0, 0)
    #signal_features = env.df.loc[:, ['Close', 'Open', 'High', 'Low','Volume']].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Close', 'Volume']].to_numpy()[start:end]
    signal_features = np.column_stack((prices, diff,signal_features))

    return prices, signal_features

class MyStockEnv(StocksEnv):
    _process_data = my_process_data

env = MyStockEnv(
               df = STOCKS_GOOGL,
               window_size = 10,
               #frame_bound = (10, 300)
               frame_bound = (10, len(STOCKS_GOOGL)))
print()
print("custom_env information:")
print("> shape:", env.shape)
print("> df.shape:", env.df.shape)
print("> prices.shape:", env.prices.shape)
print("> signal_features.shape:", env.signal_features.shape)
print("> max_possible_profit:", env.max_possible_profit())
print("> trade_fee_bid_percent:",env.trade_fee_bid_percent)
print("> trade_fee_ask_percent:",env.trade_fee_ask_percent)

observation = env.reset()

observation, reward, done, info = env.step(1)
print("x",info,reward)
observation, reward, done, info = env.step(1)
print("x",info,reward)
observation, reward, done, info = env.step(0)
print("x",info,reward)
observation, reward, done, info = env.step(0)
print("x",info,reward)
observation, reward, done, info = env.step(1)
print("x",info,reward)
observation, reward, done, info = env.step(0)
print("x",info,reward)

# i = 0
# while True:
#     action = env.action_space.sample()
#     observation, reward, done, info = env.step(action)
#     if i==0:
#         env.render()
#     print(action,info,reward)
#     if done:
#         print(i)
#         print("info:",info)
#         break
#     i+=1
# plt.cla()
# env.render_all()
# plt.show()

class Model(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(Model, self).__init__()
        hid1_size = act_dim * 10
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, act_dim)

    def forward(self, x):
        out = paddle.tanh(self.fc1(x))
        prob = F.softmax(self.fc2(out), axis=-1)
        return prob

class Agent(parl.Agent):
    def __init__(self, algorithm):
        super(Agent, self).__init__(algorithm)

    def sample(self, obs):
        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        prob = prob.numpy()
        act = np.random.choice(len(prob), 1, p=prob)[0]
        return act

    def predict(self, obs):
        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        act = prob.argmax().numpy()[0]
        return act

    def learn(self, obs, act, reward):
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)

        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')

        loss = self.alg.learn(obs, act, reward)
        return loss.numpy()[0]

def run_train_episode(agent, env, render=False):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs = data_process(obs)
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)
        if render:
            env.render()
        if done:
            break
    return obs_list, action_list, reward_list

def run_evaluate_episodes(agent, env, eval_episodes=1, render=False):
    eval_reward = []
    for i in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = data_process(obs)
            action = agent.predict(obs)
            obs, reward, isOver, info = env.step(action)
            print(action,info,reward)
            episode_reward += reward
            if isOver:
                print("info:",info)
                break
        if render:
            plt.cla()
            env.render_all()
            plt.show()
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

def data_process(obs):
    # 缩放数据尺度
    obs[:,0]/=100
    return obs.astype(np.float).ravel()

def calc_reward_to_go(reward_list, gamma=0.99):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)

obs_dim = env.observation_space.shape[0]*env.observation_space.shape[1]
act_dim = env.action_space.n
model = Model(obs_dim=obs_dim, act_dim=act_dim)
alg = parl.algorithms.PolicyGradient(model, lr=1e-3)
agent = Agent(alg)
logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

if os.path.exists('./trade.ckpt'):
    agent.restore('./trade.ckpt')
    logger.info("Total reward: {}".format(run_evaluate_episodes(agent, env, render=True)))
    exit()

for i in range(1000):
    obs_list, action_list, reward_list = run_train_episode(agent, env,render=False)
    if i % 10 == 0:
        logger.info("Episode {}, Reward Sum {}.".format(
            i, sum(reward_list)))

    batch_obs = np.array(obs_list)
    batch_action = np.array(action_list)

    reward_list = calc_reward_to_go(reward_list)

    agent.learn(batch_obs, batch_action, reward_list)
    if (i + 1) % 100 == 0:
        total_reward = run_evaluate_episodes(agent, env, render=False)
        logger.info('Test reward: {}'.format(total_reward))
agent.save('./trade.ckpt')
