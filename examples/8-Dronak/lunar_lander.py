#!/usr/bin/env python3

# Usage: Launch from the root directory of the repository
# $ PYTHONPATH=. python examples/8-Dronak/lunar_lander.py

import gymnasium as gym
from SimpleBaselines.agent.rl_agents.DeterministicDQN_RL_Agent import DeterministicDQN_RL_Agent
from SimpleBaselines.states.State import State
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
import time
import sys
import os
import glob
import cv2


# Plot styling
plt.style.use('ggplot')

# Hyperparameters
num_episodes = 500
seed = 42

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

# experiment results
rewards_total = list()
steps_total = list()
egreedy_total = list()
solved_after = 0
start_time = time.time()

# Reporting parameters
report_interval = 40
STEPS_TO_SOLVE = 195

# Create the agent
agent = DeterministicDQN_RL_Agent(
    env=env,
    seed=seed,
    gamma=0.99,
    nn_learning_rate=0.01,
    egreedy=0.95,
    egreedy_final=0.001,
    egreedy_decay=1e3,
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
            print("Average Reward [last {}]: {:.2f}".format(report_interval,
                                                            sum(rewards_total[-report_interval:]) / report_interval))
            print("Average Reward [last 100]: {:.2f}".format(sum(rewards_total[-100:]) / 100))
            print("Average Reward: {:.2f}".format(sum(rewards_total) / len(steps_total)))

            print("Average Steps [last {}]: {:.2f}".format(report_interval,
                                                           sum(steps_total[-report_interval:]) / report_interval))
            print("Average Steps [last 100]: {:.2f}".format(sum(steps_total[-100:]) / 100))
            print("Average Steps: {:.2f}".format(sum(steps_total) / len(steps_total)))

            print("Epsilon: {:.2f}".format(agent.egreedy))
            print("Frames Total: {}".format(agent.current_state.step))
            print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

            # Play test on env_test
            observation, info = env_test.reset(seed=seed)
            state = State(observation=observation, info=info)
            while not (state.terminated or state.truncated):
                action = agent.__action_decision_function__(state)
                observation, reward, terminated, truncated, info = env_test.step(action)
                state = State(observation=observation, terminated=terminated, truncated=truncated, info=info)

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

env.close()
env_test.close()

plt.figure(figsize=(12,5))
plt.title("Cumulative Rewards")
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