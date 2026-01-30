import optuna
import gymnasium as gym
from SimpleBaselines.agent.rl_agents.DuelingDetDQN_RL_Agent import DuelingDetDQN_RL_Agent
from SimpleBaselines.states.State import State
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time

def objective(trial):
    # Hyperparameter suggestions from Optuna
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    nn_learning_rate = trial.suggest_float("nn_learning_rate", 1e-5, 1e-2, log=True)
    egreedy = trial.suggest_float("egreedy", 0.8, 1.0)
    egreedy_final = trial.suggest_float("egreedy_final", 0.001, 0.05)
    egreedy_decay = trial.suggest_int("egreedy_decay", 500, 5000)
    hidden_layers_size = trial.suggest_categorical("hidden_layers_size", [[64], [128], [256], [64, 64], [128, 128], [256, 256], [64, 64, 64], [128, 128, 128]])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    #use_batch_norm = trial.suggest_categorical("use_batch_norm", [False, True]) batch normalization not working
    memory_size = trial.suggest_int("memory_size", 1000, 10000)
    batch_size = trial.suggest_int("batch_size", 16, 128)
    target_net_update_steps = trial.suggest_int("target_net_update_steps", 100, 1000)

    # Training environment
    env = gym.make(
        "LunarLander-v3",
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5
    )
    # Create the agent
    agent = DuelingDetDQN_RL_Agent(
        env=env,
        seed=42,
        gamma=gamma,
        nn_learning_rate=nn_learning_rate,
        egreedy=egreedy,
        egreedy_final=egreedy_final,
        egreedy_decay=egreedy_decay,
        hidden_layers_size=hidden_layers_size,
        activation_fn=nn.Tanh,
        dropout=dropout,
        use_batch_norm=False,
        loss_fn=nn.MSELoss,
        optimizer=optim.Adam,
        memory_size=memory_size,
        batch_size=batch_size,
        target_net_update_steps=target_net_update_steps,
        clip_error=True
    )

    # Training loop
    num_episodes = 500  # Use a lower number for faster evaluation
    cumulative_rewards = []

    for episode in range(num_episodes):
        agent.reset_env(seed=42)
        agent.play(max_steps=1000, seed=42)

        if agent.current_state.terminated or agent.current_state.truncated:
            cumulative_rewards.append(agent.final_state.cumulative_reward)

    # Return average reward as the objective value
    env.close()
    return np.mean(cumulative_rewards)

# Optuna study setup
study = optuna.create_study(direction="maximize", study_name="dueling_dqn")
study.optimize(objective, n_trials=100)  # Number of trials can be adjusted

# Print best hyperparameters
print("Best hyperparameters:", study.best_params)

# Load the best hyperparameters from json file
with open("best_params/best_params.json", "r") as file:
    best_params = json.load(file)

best_params["dueling_dqn"] = study.best_params

# Update and save hyperparameters
with open("best_params/best_params.json", "w", encoding= "utf-8") as file:
    json.dump(best_params, file, indent=4, ensure_ascii=False)


