#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gymnasium as gym
from SimpleBaselines.agent.rl_agents.DeterministicDQN_RL_Agent import DeterministicDQN_RL_Agent
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Plot styling
plt.style.use('ggplot')

# Create the environment
env = gym.make('CartPole-v1')


# Analyze the environment
print('Observation space:', env.observation_space)
print('Observation space sample:', env.observation_space.sample())
print('Action space:', env.action_space)
print('Action space sample:', env.action_space.sample())

# experiment hyperparameters
num_episodes = 2000
seed = 42

# experiment results
rewards_total = list()
steps_total = list()

# Create the agent
agent = DeterministicDQN_RL_Agent(
    env=env,
    seed=seed,
    gamma=0.99,
    nn_learning_rate=0.025,
    egreedy=0.9,
    egreedy_final=0.01,
    egreedy_decay=0.75,
    hidden_layers_size=[64],
    activation_fn=nn.Tanh,
    dropout=0.0,
    use_batch_norm=False,
    loss_fn=nn.MSELoss,
    optimizer=optim.Adam
)

# Play the game
for episode in range(num_episodes):

    agent.reset_env(seed=seed)

    # Play the game
    agent.play(max_steps=5000, seed=seed)

    # Print some reporting
    #agent.reporting.print_short_report()
    #agent.reporting.print_report()

    rewards_total.append(agent.final_state.cumulative_reward)
    steps_total.append(agent.final_state.step + 1)

# Close the environment
env.close()


rewards_total = np.array(rewards_total)
print("Percent of episodes finished successfully: {}".format(sum(rewards_total > 195) / num_episodes))
print("Percent of episodes finished successfully (last 100 episodes): {}".format(sum(rewards_total[-100:] > 195) / 100))

print("Average number of steps: {}".format(sum(steps_total)/num_episodes))
print("Average number of steps (last 100 episodes): {}".format(sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green', width=5)
plt.show()

plt.figure(figsize=(12,5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red', width=5)
plt.show()