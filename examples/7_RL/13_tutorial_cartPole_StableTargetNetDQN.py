#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
from SimpleBaselines.agent.rl_agents.StableTargetNetDetDQN_RL_Agent import StableTargetNetDetDQN_RL_Agent
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import time
import sys


# Plot styling
plt.style.use('ggplot')

# Hyperparameters
num_episodes = 500
seed = 42

# Create the environment
env = gym.make('CartPole-v1')

# Analyze the environment
print('Observation space:', env.observation_space)
print('Observation space sample:', env.observation_space.sample())
print('Action space:', env.action_space)
print('Action space sample:', env.action_space.sample())

# experiment results
rewards_total = list()
steps_total = list()
egreedy_total = list()
solved_after = 0
start_time = time.time()

# Reporting parameters
report_interval = 10
STEPS_TO_SOLVE = 195

# Create the agent
agent = StableTargetNetDetDQN_RL_Agent(
    env=env,
    seed=seed,
    gamma=0.99,
    nn_learning_rate=0.01,
    egreedy=0.9,
    egreedy_final=0.02,
    egreedy_decay=500,
    hidden_layers_size=[64],
    activation_fn=nn.Tanh,
    dropout=0.0,
    use_batch_norm=False,
    loss_fn=nn.MSELoss,
    optimizer=optim.Adam,
    memory_size=50000,
    batch_size=32,
    target_net_update_steps=100,
    clip_error=False
)


for episode in range(num_episodes):

    agent.reset_env(seed=seed)

    # Play the game
    agent.play(max_steps=5000, seed=seed)

    # Print some reporting
    # agent.reporting.print_short_report()
    # agent.reporting.print_report()

    if agent.current_state.terminated or agent.current_state.truncated:
        steps_total.append(agent.current_state.step )
        mean_reward_100 = sum(steps_total[-100:]) / min(len(steps_total), 100)
        rewards_total.append(agent.final_state.cumulative_reward)
        egreedy_total.append(agent.egreedy)

        if mean_reward_100 > STEPS_TO_SOLVE and solved_after == 0:
            print("*************************")
            print("SOLVED! After {} episodes".format(episode))
            print("*************************")
            solved_after = episode

        if episode % report_interval == 0:
            elapsed_time = time.time() - start_time
            print("-----------------")
            print("Episode: {}".format(episode))
            print("Average Reward [last {}]: {:.2f}".format(report_interval, sum(rewards_total[-report_interval:]) / report_interval))
            print("Average Reward [last 100]: {:.2f}".format(sum(rewards_total[-100:]) / 100))
            print("Average Reward: {:.2f}".format(sum(rewards_total) / len(steps_total)))

            print("Average Steps [last {}]: {:.2f}".format(report_interval, sum(steps_total[-report_interval:]) / report_interval))
            print("Average Steps [last 100]: {:.2f}".format(sum(steps_total[-100:]) / 100))
            print("Average Steps: {:.2f}".format(sum(steps_total) / len(steps_total)))

            print("Epsilon: {:.2f}".format(agent.egreedy))
            print("Frames Total: {}".format(agent.current_state.step))
            print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")


# Close the environment
env.close()

# Print the results
if solved_after > 0:
    print("Solved after {} episodes".format(solved_after))
else:
    print("Could not solve after {} episodes".format(num_episodes))

rewards_total = np.array(rewards_total)
steps_total = np.array(steps_total)
egreedy_total = np.array(egreedy_total)

print("Average reward: {}".format( sum(rewards_total) / num_episodes) )
print("Average reward (last 100 episodes): {}".format( sum(rewards_total[-100:]) / 100) )
print("Percent of episodes finished successfully: {}".format( sum(rewards_total > STEPS_TO_SOLVE) / num_episodes) )
print("Percent of episodes finished successfully (last 100 episodes): {}".format( sum(rewards_total[-100:] > STEPS_TO_SOLVE) / 100) )
print("Average number of steps: {}".format( sum(steps_total)/num_episodes) )
print("Average number of steps (last 100 episodes): {}".format( sum(steps_total[-100:])/100) )

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(rewards_total)), rewards_total, alpha=0.6, color='green', width=5)
plt.show()

plt.figure(figsize=(12,5))
plt.title("Steps / Episode length")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='red', width=5)
plt.show()

plt.figure(figsize=(12,5))
plt.title("Egreedy / Episode length")
plt.bar(torch.arange(len(egreedy_total)), egreedy_total, alpha=0.6, color='red', width=5)
plt.show()