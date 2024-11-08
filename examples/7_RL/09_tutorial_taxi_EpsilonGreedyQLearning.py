#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from SimpleBaselines.agent.rl_agents.EpsilonGreedyQLearning_RL_Agent import EpsilonGreedyQLearning_RL_Agent

# Plot styling
plt.style.use('ggplot')

# Create the environment
env = gym.make('Taxi-v3')

# Analyze the environment
print('Observation space:', env.observation_space)
print('Observation space sample:', env.observation_space.sample())
print('Action space:', env.action_space)
print('Action space sample:', env.action_space.sample())

# experiment hyperparameters
num_episodes = 1000
seed = 42

# experiment results
rewards_total = list()
steps_total = list()

# Create the agent
agent = EpsilonGreedyQLearning_RL_Agent(env=env, seed=seed, gamma=0.98, learning_rate=0.85, egreedy=0.05, egreedy_final=0.01, egreedy_decay=0.999)

# Play the game
for episode in range(num_episodes):

    agent.reset_env(seed=seed)

    # Play the game (new environment for each run with continuously learning agent)
    agent.play(max_steps=5000, seed=seed)

    # Print some reporting
    # ValueIteration_agent.reporting.print_short_report()
    # ValueIteration_agent.reporting.print_report()

    rewards_total.append(agent.final_state.cumulative_reward)
    steps_total.append(agent.final_state.step + 1)

# Close the environment
env.close()

print("Percent of episodes finished successfully: {}".format(sum(rewards_total)/num_episodes))
print("Percent of episodes finished successfully (last 100 episodes): {}".format(sum(rewards_total[-100:])/100))

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