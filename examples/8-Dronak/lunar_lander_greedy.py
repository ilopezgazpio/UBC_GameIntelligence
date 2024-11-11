#!/usr/bin/env python3

import gymnasium as gym
import matplotlib.pyplot as plt
import torch

## import the agent from ../../SimpleBaselines/agent/rl_agents/Random_RL_Agent.py
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR + "/../../")
from SimpleBaselines.agent.rl_agents.EpsilonGreedyQLearning_RL_Agent import EpsilonGreedyQLearning_RL_Agent

env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)


# experiment hyperparameters
num_episodes = 1000
seed = 42

# experiment results
rewards_total = list()
steps_total = list()

# Create the agent
agent = EpsilonGreedyQLearning_RL_Agent(env, seed=seed)

# Play the game
for episode in range(num_episodes):

    agent.reset_env(seed=seed)

    # Play the game (new environment for each run with continuously learning agent)
    agent.play(max_steps=5000, seed=seed)

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
