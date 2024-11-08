#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gymnasium as gym
import torch
import matplotlib.pyplot as plt
from SimpleBaselines.agent.rl_agents.Random_RL_Agent import Random_RL_Agent

# Create the environment
env = gym.make('FrozenLake-v1')

# Analyze the environment
print('Observation space:', env.observation_space)
print('Observation space sample:', env.observation_space.sample())
print('Action space:', env.action_space)
print('Action space:', env.action_space.sample())

# experiment hyperparameters
num_episodes = 1000
seed = 42

# experiment results
rewards_total = list()
steps_total = list()

# Play the game
for episode in range(num_episodes):

    # Create the agent
    random_agent = Random_RL_Agent(env)

    # Play the game
    random_agent.play(max_steps=5000, seed=seed)

    # Print some reporting
    random_agent.reporting.print_short_report()
    random_agent.reporting.print_report()

    rewards_total.append(random_agent.final_state.cumulative_reward)
    steps_total.append(random_agent.final_state.step + 1)

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

'''
It is also possible to save videos of the agent with gymnasium.wrappers.Monitor

example:

videosDir = './RLvideos/'
env = gym.make('CartPole-v1')
env = gym.wrappers.Monitor(env, videosDir)

If you find errors in the process, check attributes of the environment with:

# import site
# site.getsitepackages() 

and check into gymnasium / env folders for attributes or settings

'''
