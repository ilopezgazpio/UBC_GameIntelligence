#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
from SimpleBaselines.agent.rl_agents.DeterministicDQN_RL_Agent import DeterministicDQN_RL_Agent
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
agent = DeterministicDQN_RL_Agent(
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
    optimizer=optim.Adam
)


for episode in range(num_episodes):

    agent.reset_env(seed=seed)

    # Play the game
    # agent.play(max_steps=5000, seed=seed)

    # Print some reporting
    # agent.reporting.print_short_report()
    # agent.reporting.print_report()
    while True:
        agent.current_state.step += 1
        agent.update_egreedy()
        action = agent.__DQN_decision_function__(agent.current_state)
        new_state, reward, terminated, truncated, _ = agent.env.step(action)
        agent.__DQN_bellman_update__(agent.current_state, action, new_state, reward, terminated, truncated)

        # Update current state
        agent.current_state.reward = reward
        agent.current_state.terminated = terminated
        agent.current_state.truncated = truncated
        agent.current_state.action = action
        agent.current_state.action_history.append(action)
        agent.current_state.cumulative_reward += reward
        agent.current_state.observation = new_state

        if terminated or truncated:
            steps_total.append(agent.current_state.step )
            #rewards_total.append(agent.final_state.cumulative_reward)
            #steps_total.append(agent.final_state.step + 1)
            #egreedy_total.append(agent.egreedy)
            mean_reward_100 = sum(steps_total[-100:]) / min(len(steps_total), 100)

            if mean_reward_100 > STEPS_TO_SOLVE and solved_after == 0:
                print("*************************")
                print("SOLVED! After {} episodes".format(episode))
                print("*************************")
                solved_after = episode

            if episode % report_interval == 0:
                elapsed_time = time.time() - start_time
                print("-----------------")
                print("Episode: {}".format(episode))
                print("Average Reward [last {}]: {:.2f}".format(report_interval, sum(steps_total[-report_interval:]) / report_interval))
                print("Average Reward [last 100]: {:.2f}".format(sum(steps_total[-10:]) / 100))
                print("Average Reward: {:.2f}".format(sum(steps_total) / len(steps_total)))
                print("Epsilon: {:.2f}".format(agent.egreedy))
                print("Frames Total: {}".format(agent.current_state.step))
                print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")
            break


# Close the environment
env.close()


if solved:
    print("Solved after {} episodes".format(solved_after))
else:
    print("Could not solve after {} episodes".format(num_episodes))

rewards_total = np.array(rewards_total)
steps_total = np.array(steps_total)
egreedy_total = np.array(egreedy_total)

print("Average reward: {}".format( sum(steps_total) / num_episodes) )
print("Average reward (last 100 episodes): {}".format( sum(steps_total[-100:]) / 100) )
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