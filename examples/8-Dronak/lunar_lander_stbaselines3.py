#!/usr/bin/env python3

# Usage: Launch from the root directory of the repository
# $ PYTHONPATH=. python examples/8-Dronak/lunar_lander.py

import gymnasium as gym
import stable_baselines3 as sb3
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict


params = OrderedDict([('batch_size', 128),
             ('buffer_size', 50000),
             ('exploration_final_eps', 0.1),
             ('exploration_fraction', 0.12),
             ('gamma', 0.99),
             ('gradient_steps', -1),
             ('learning_rate', 0.00063),
             ('learning_starts', 0),
             ('policy', 'MlpPolicy'),
             ('policy_kwargs', 'dict(net_arch=[256, 256])'),
             ('target_update_interval', 250),
             ('train_freq', 4)])

parameters = {
    'batch_size': 128,
    'buffer_size': 50000,
    'exploration_final_eps': 0.1,
    'exploration_fraction': 0.12,
    'gamma': 0.99,
    'gradient_steps': -1,
    'learning_rate': 0.00063,
    'learning_starts': 0,
    'policy': 'MlpPolicy',
    'policy_kwargs': dict(net_arch=[256, 256]),
    'target_update_interval': 250,
    'train_freq': 4
}

# Hyperparameters
n_timesteps = 100000
eval_episodes = 200
eval_timesteps = 150000

# Create the training environment
env = gym.make(
    "LunarLander-v3",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5
)

# Create the testing environment
env_test = gym.make(
    "LunarLander-v3",
    continuous=False,
    gravity=-10.0,
    enable_wind=False,
    wind_power=15.0,
    turbulence_power=1.5,
    render_mode='human'
)

# Analyze the environment
print('Observation space:', env.observation_space)
print('Observation space sample:', env.observation_space.sample())
print('Action space:', env.action_space)
print('Action space sample:', env.action_space.sample())

# Create the agent
agent = sb3.DQN(env = env, **parameters)

# Train the agent
agent.learn(n_timesteps)

# Test the agent
agent.set_env(env_test)
rewards = []
for _ in range(eval_episodes):
    obs, info = env_test.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = agent.predict(obs)
        obs, reward, done, _, info = env_test.step(action)
        episode_reward += reward
    rewards.append(episode_reward)

# Close the environments
env.close()
env_test.close()

# Show mean rewards
print('Mean reward:', np.mean(rewards))

# Plot the rewards
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards')
plt.show()