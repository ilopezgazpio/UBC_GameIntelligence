import matplotlib.pyplot as plt
plt.style.use('ggplot')
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


"""
Crear enviroment de entrenamiento
"""

import gymnasium as gym

env = gym.make("LunarLander-v3",
               continuous = False,
               gravity = -10.0,
               enable_wind = False,
               wind_power = 0.0,
               turbulence_power = 0.0)


# Analizar el environment
print('Observation space: ', env.observation_space)
print('Observation space sample: ', env.observation_space.sample())
print('Action space: ', env.action_space) # 0: nada, 1: principal, 2: izquierdo, 3: derecho
print('Action space sample: ', env.action_space.sample())




"""
Crear el agente RL
"""

from SimpleBaselines.agent.rl_agents.DeterministicDQN_RL_Agent import DeterministicDQN_RL_Agent
from SimpleBaselines.states.State import State

num_episodes = 500
max_steps = 10000
seed = 42

rewards_total = list()
steps_total = list()
egreedy_total = list()
solved_after = 0 # para saber en qué episodio se ha resuelto (o no, 0)
MIN_MEAN_REWARDS = 200 # condición de resuelto
start_time = time.time()
report_interval = 50



agent = DeterministicDQN_RL_Agent(
    env = env,
    seed = seed,
    gamma = 0.99, # descuento futuro
    nn_learning_rate = 0.01, # tasa alta de aprendizaje
    egreedy = 0.95, # empieza explorando mucho, 95% de acciones aleatorias
    egreedy_final = 0.001, # acaba casi explotando completamente, acciones óptimas conocidas
    egreedy_decay = 35e3, # desciende lentamente para explorar y terminar explotando
    hidden_layers_size = [64],
    activation_fn = nn.Tanh,
    dropout = 0.0,
    use_batch_norm = False,
    loss_fn = nn.MSELoss,
    optimizer = optim.Adam
)


for episode in range(num_episodes):

  agent.reset_env(seed = seed)

  # Play game
  agent.play(max_steps = max_steps, seed = seed)

  # Comprobar si ha terminado de manera natural (aterrizado, estrellado) o de manera artificial (max_steps)
  if agent.current_state.terminated or agent.current_state.truncated:
    steps_total.append(agent.current_state.step) # cuántos pasos ha hecho en el estado actual
    rewards_total.append(agent.final_state.cumulative_reward) # guardar la recompensa final obtenida
    egreedy_total.append(agent.egreedy) # valor actual de epsilon
    mean_reward_100 = sum(rewards_total[-100:]) / min(len(rewards_total), 100) # media de las últimas 100 recompensas

    # Condición de resuelto
    if mean_reward_100 > MIN_MEAN_REWARDS and solved_after == 0:
      print('************************')
      print('SOLVED! After {} episodes'.format(episode))
      print('************************')
      solved_after = episode + 1 # porque empieza en episode = 0

    # Hacer reporte cada 50 episodios
    if episode % report_interval == 0 and episode != 0:
      elapsed_time = time.time() - start_time
      print('----------------')
      print('Episode: {}'.format(episode))
      print('Average Reward [last {}]: {:.2f}'.format(report_interval,
                                                     np.mean(rewards_total[-report_interval:]))) # average reward de los ultimos episodes
      print('Average Reward [last 100]: {:.2f}'.format(np.mean(rewards_total[-100:]))) # average reward de los ultimos 100
      print('Average Reward: {:.2f}'.format(np.mean(rewards_total)))

      # mismo reporte que las recompensas pero con pasos
      print("Average Steps [last {}]: {:.2f}".format(report_interval,
                                                           np.mean(steps_total[-report_interval:])))
      print("Average Steps [last 100]: {:.2f}".format(np.mean(steps_total[-100:])))
      print("Average Steps: {:.2f}".format(np.mean(steps_total)))

      print("Epsilon: {:.2f}".format(agent.egreedy))
      print("Episode steps: {}".format(agent.current_state.step))
      print("Frames total: {}".format(sum(steps_total)))
      print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}")

# Print the results
if solved_after > 0:
  print('Solved after {} episodes'.format(solved_after))
else:
  print('Could not solve after {} episodes'.format(num_episodes))

env.close()

rewards_total = np.array(rewards_total)
steps_total = np.array(steps_total)
egreedy_total = np.array(egreedy_total)

print("\n")
print("Average reward: {}".format(np.mean(rewards_total)))
print("Average reward (last 100 episodes): {}".format(np.mean(rewards_total[-100:])))
print("Percent of episodes finished successfully: {}%".format(sum(rewards_total > MIN_MEAN_REWARDS)/num_episodes*100))
print("Percent of episodes finished successfully (last 100 episodes): {}%".format(sum(rewards_total[-100:] > MIN_MEAN_REWARDS)))
print("Average number of steps: {}".format(np.mean(steps_total)))
print("Average number of steps (last 100 episodes): {}".format(np.mean(steps_total[-100:])) )
