import gymnasium as gym
import json
import datetime as dt
import pandas as pd
from stable_baselines3 import PPO


# The algorithms require a vectorized environment to run
env = gym.make('gym_StockTrading:StockTrading-v0')

# Instantiate the PPO model
model = PPO('MlpPolicy', env, verbose=1)

# Train the model
model.learn(total_timesteps=20000)

# Reset the environment
obs, _ = env.reset()

# Run the model for 2000 steps
for i in range(2000):
    action, _states = model.predict(obs)

    # Perform a step in the environment
    obs, rewards, terminated, truncated, info = env.step(action)

    # Check if the episode is over (terminated or truncated)
    if terminated or truncated:
        obs, _ = env.reset()  # Reset the environment if the episode is over

    # Render the environment
    env.render()