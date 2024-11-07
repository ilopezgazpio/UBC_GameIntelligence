#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
from SimpleBaselines.agent.rl_agents.Random_RL_Agent import Random_RL_Agent

# Create the environment
env = gym.make('CartPole-v1')

# Analyze the environment
print('Observation space:', env.observation_space)
print('Observation space sample:', env.observation_space.sample())
print('Action space:', env.action_space)
print('Action space:', env.action_space.sample())

# experiment hyperparameters
num_episodes = 1
seed = 42

# Play the game
for episode in range(num_episodes):

    # Create the agent
    random_agent = Random_RL_Agent()

    # Play the game
    random_agent.play(environment=env, max_steps=5000, seed=seed)

    # Print some reporting
    random_agent.reporting.print_short_report()
    random_agent.reporting.print_report()

# Close the environment
env.close()



