import gym
env = gym.make('CartPole-v0')
#env = gym.make('MountainCar-v0')
#env = gym.make('MsPacman-v0')
for i_episode in range(1):
    observation = env.reset()
    #for t in range(100):
    t = 0;
    totalReward = 0;
    while True:
        env.render()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        totalReward += reward;
        if done:
            print("Episode finished after {} timesteps. Reward {}.".format(t+1,totalReward))
            break
        t+=1;
env.close()
