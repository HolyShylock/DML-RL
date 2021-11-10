import gym

from BespokeAgent import BespokeAgent
env = gym.make('MountainCar-v0')
print('observation space = {}'. format(env.observation_space))
print('action space = {}'.format(env.action_space))
print('observation scope = {} ~ {}'.format(env.observation_space.low, env.observation_space.high))
print('action number = {}'.format(env.action_space.n))

agent = BespokeAgent(env)

def play(env, agent, render = False, train = False):
    episode_reward = 0
    observation = env.reset()
    while True:
        if render:
            env.render()
        action = agent.decide(observation)
        next_observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn(observation, action, reward, done)
        if done:
            break
        observation = next_observation
    return episode_reward

env.seed(0)
episode_reward = play(env, agent, render = True)
print('reward = {}'.format(episode_reward))
env.close()