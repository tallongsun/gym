import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import os
import gym
import numpy as np
import parl
from parl.utils import logger
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

LEARNING_RATE = 1e-3
TYPE = "car"

class Model(parl.Model):
    """ Linear network to solve Cartpole problem.

    Args:
        obs_dim (int): Dimension of observation space.
        act_dim (int): Dimension of action space.
    """

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
    """Agent of Cartpole env.

    Args:
        algorithm(parl.Algorithm): algorithm used to solve the problem.

    """

    def __init__(self, algorithm):
        super(Agent, self).__init__(algorithm)

    def sample(self, obs):
        """Sample an action when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)

        Returns:
            act(int): action
        """
        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        prob = prob.numpy()
        act = np.random.choice(len(prob), 1, p=prob)[0]
        return act

    def predict(self, obs):
        """Predict an action when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)

        Returns:
            act(int): action
        """
        obs = paddle.to_tensor(obs, dtype='float32')
        prob = self.alg.predict(obs)
        act = prob.argmax().numpy()[0]
        return act

    def learn(self, obs, act, reward):
        """Update model with an episode data

        Args:
            obs(np.float32): shape of (batch_size, obs_dim)
            act(np.int32): shape of (batch_size)
            reward(np.float32): shape of (batch_size)

        Returns:
            loss(float)

        """
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)

        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')

        loss = self.alg.learn(obs, act, reward)
        return loss.numpy()[0]

def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    if TYPE=="pong":
        image = image[35:195]  # 裁剪
    image = image[::2, ::2, 0]  # 下采样，缩放2倍
    image[image == 144] = 0  # 擦除背景 (background type 1)
    image[image == 109] = 0  # 擦除背景 (background type 2)
    image[image != 0] = 1  # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float).ravel()

# train an episode
def run_train_episode(agent, env, render=False,prepro=False):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        preHeight = obs[0]
        maxHeight = -0.45

        if prepro:
            obs = preprocess(obs)
        obs_list.append(obs)
        action = agent.sample(obs)
        action_list.append(action)

        obs, reward, done, info = env.step(action)

        if TYPE == "car":
            if obs[0] > preHeight:
                reward += 1
            if obs[0] > maxHeight:
                maxHeight = obs[0]
                reward += 2

        reward_list.append(reward)
        if render:
            env.render()
        if done:
            break
    return obs_list, action_list, reward_list


# evaluate 5 episodes
def run_evaluate_episodes(agent, env, eval_episodes=1, render=False,prepro=False):
    eval_reward = []
    for i in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        while True:
            if prepro:
                obs = preprocess(obs)
            action = agent.predict(obs)
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def calc_reward_to_go(reward_list, gamma=0.99):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    reward_list -= np.mean(reward_list)
    reward_list /= np.std(reward_list)
    return np.array(reward_list)

#Function that apply the algorithm for future discount reward
def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = cumulative_rewards * discount_rate + rewards[step]
        discounted_rewards[step] = cumulative_rewards

    return discounted_rewards

# This function apply the discount_rewards for every game in the n_game_per_iter, then normalize it
def discount_and_normalized_rewards(all_rewards, discount_rate=0.95):
    all_discount_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flatten = np.concatenate(all_discount_rewards)
    mean = flatten.mean()
    std = flatten.std()
    return [(discounted_rewards - mean)/std for discounted_rewards in all_discount_rewards]

def create_env():
    if TYPE == "mario":
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        return env
    elif TYPE == "car":
        env = gym.make('MountainCar-v0')
        return env
    elif TYPE == "pong":
        env = gym.make('Pong-v0')
        return env
    else:
        env = gym.make('CartPole-v0')
        return env




def main():
    prepro = False
    env = create_env()
    # env = env.unwrapped # Cancel the minimum score limit
    if TYPE == "pong":
        obs_dim = 80*80
        prepro = True
    elif TYPE == "mario":
        obs_dim = 120*128
        prepro = True
    else:
        obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # build an agent
    model = Model(obs_dim=obs_dim, act_dim=act_dim)
    alg = parl.algorithms.PolicyGradient(model, lr=LEARNING_RATE)
    agent = Agent(alg)

    # load model and evaluate
    if os.path.exists('./'+TYPE+'.ckpt'):
        agent.restore('./'+TYPE+'.ckpt')
        logger.info("Total reward: {}".format(run_evaluate_episodes(agent, env, render=True,prepro=prepro)))
        exit()

    for i in range(1000):
        obs_list, action_list, reward_list = run_train_episode(agent, env,render=False,prepro=prepro)
        if i % 10 == 0:
            logger.info("Episode {}, Reward Sum {}.".format(
                i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)

        reward_list = calc_reward_to_go(reward_list)
        # all_rewards = []
        # all_rewards.append(reward_list)
        # batch_reward = discount_and_normalized_rewards(all_rewards)

        agent.learn(batch_obs, batch_action, reward_list)
        if (i + 1) % 100 == 0:
            total_reward = run_evaluate_episodes(agent, env, render=True,prepro=prepro)
            logger.info('Test reward: {}'.format(total_reward))

    # save the parameters to ./model.ckpt
    agent.save('./'+TYPE+'.ckpt')


if __name__ == '__main__':
    main()





