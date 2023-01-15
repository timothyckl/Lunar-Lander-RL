import gymnasium as gym
from dqn import Agent


env = gym.make('LunarLander-v2', render_mode='rgb_array')
lr = 0.001
discount = 0.99
exploration_rate = 1.0
exploration_decay = 0.999
update_interval = 10
num_episodes = 1000
agent = Agent(
    env=env,
    alpha=lr,
    gamma=discount,
    epsilon=exploration_rate,
    epsilon_decay=exploration_decay,
    update_interval=update_interval
)

if __name__ == '__main__':
    rewards, exploration_rate, steps_per_episode = agent.train(num_episodes=num_episodes)