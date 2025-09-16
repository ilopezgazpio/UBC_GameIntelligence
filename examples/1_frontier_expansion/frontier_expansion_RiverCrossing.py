#---------------------------------------------------------------------------
# Frontier usage explanation using the Openai Gym RiverCrossing environment
# Search Heuristics - University of the Basque Country (UPV/EHU)
# Inigo Lopez-Gazpio
#---------------------------------------------------------------------------
#
#---------------------------------------------------------------------------


'''
Preliminary notes

Gymnasium is a toolkit for developing and comparing reinforcement learning algorithms.
It supports teaching agents everything from walking to playing games like Pong or Pinball.

In class we will use the gymnasium library with certain specifically designed environments to learn AI concepts.
Gymnasium gathers all functionality regarding the environments.
UBC_GameIntelligence library gathers all functionality regarding agents.

As a first step it is recommended to read the basic documentation of gymnasium at: https://gymnasium.farama.org/
'''


#---------------------------------------------------------------------------
# The problem "Wolf, goat, and cabbage"
# https://en.wikipedia.org/wiki/River_crossing_puzzle
# A farmer must transport a Wolf, a goat and cabbage from one side of a river to another using
# a boat which can only hold one item in addition to the farmer,
# subject to the constraints that the wolf cannot be left alone with the goat,
# and the goat cannot be left alone with the cabbage


#---------------------------------------------------------------------------
# The definition of what a state is, is provided by the environment
# In order to work, first move to the OpenAIGym_extra_environments folder and follow the instructions to install gym-RiverCrossing environment
# remember that you will have to install gym itself previously
# if everything is installed correctly the following lines should work smoothly


import gymnasium as gym

# environments are constructed using the gym.make call
env = gym.make('gym_RiverCrossing:RiverCrossing-v0')
env = env.unwrapped

# environments define the observation spaces for agents. These are defined as Openai gym spaces
env.observation_space

# In this case the observation space is a Box(0, 1, (4,), int8)
# This represents a 4 dimensional array with values in the [0,1] discrete range
env.observation_space.sample()
env.observation_space.low
env.observation_space.high

# The environment also defines the actions the agent will be able to perform
env.action_space
env.action_space.n

# The action space is also an Openai gym space, in this case it is a Discrete(4) space

# Analyze the files from the gym-RiverCrossing environment to gather insight on how the environment is constructed

# To start using an environment from Openai gym, we first need to reset it, this gives as our first observation, the initial state seen by the agent
env.reset()
first_observation = env.reset()
first_observation

# We can also render the environment :)
env.render()

env.state

# As mentioned, the very first observation is defined by a 4-dimensional numpy array
# 1st position: location of the farmer  (0: left / 1: right)
# 2nd position: location of the wolf    (0: left / 1: right)
# 3rd position: location of the goat    (0: left / 1: right)
# 4th position: location of the cabbage (0: left / 1: right)

# As an initial exercise, respond to the following questions,
# 1. which method is used to check if an observation is terminal ?
# 2. how is the method implemented ?

env.episode_terminated()

# we can also hack the environment as we understand all of the details :)
import numpy as np
env.state = np.array([1,1,1,1],dtype=np.int8)
env.episode_terminated()

# Can you imagine other ways to implement the __episode_terminated__ method ?


#---------------------------------------------------------------------------
# Actions: action space, conditions and effects
# 4 different actions can be done
# 0: "farmer"  -> Moves farmer alone to the other side of the river
# 1: "wolf"    -> Moves farmer and wolf
# 2: "goat"    -> Moves farmer and goat
# 3: "cabbage" -> Moves farmer and cabbage

# 0 move: moving the farmer can be done if no restrictions are violated
# 1-3 moves: moving another element can be done if no restrictions are violated and the element is in the same side of the river than the farmer

# analyze the environment's python implementation and figure out what are the restrictions implemented
# note that officially in openai gym agent movements are performed using the env.step() function
# here, we define various auxiliar methods __is_applicable__(action) and __do_step__(action) to facilitate the usage of the environment

# So, we can check if moving the goat can be done as:
env.is_applicable(2)

# Or moving the cabbage (wolf and goat cannot be in similar place)
env.is_applicable(3)

# or moving the wolf
env.is_applicable(1)

# And the effect of applying an action (goat, for instance) over a state is inverting the corresponding positions:
env.step(2)

# Openai gym is originally built for reinforcement learning agents, that is why each step produces 4 values
env.reset()
env.render()
new_observation, reward, terminated, truncated, info = env.step(2)
[new_observation, reward, terminated, truncated, info]

new_observation, reward, terminated, truncated, info = env.step(3)
[new_observation, reward, terminated, truncated, info]


# So once we have analyzed the environment in detail, it is time to start exploring the UDAI library
# We will start analyzing the method to expand nodes
# the node class is a wrapper for the environment to save extra information

env.reset()
from SimpleBaselines.frontier.Node import Node
first_node = Node(env)

# We have our initial node to start the search process
# We can also call openai gym commands through the node env variable, as the environment is saved there :)
first_node.env.render()
first_node.env.state


# Define a frontier with nodes pending of "expansion"
from SimpleBaselines.frontier.Frontier import Frontier
frontier = Frontier()
frontier.nodes.append(first_node)

# For the first node in the frontier, we have to extract it and check if it is a final state.
# if not, we have to check which of the actions are applicable

current_node = frontier.nodes.popleft()
current_node.env.render()
current_node.terminated or current_node.truncated

from SimpleBaselines.agent.AbstractTreeGraphAgent import AbstractTreeGraphAgent
searchAgent = AbstractTreeGraphAgent()
first_level_expansion_nodes = searchAgent.__expand_node__(current_node)

# analyze the function expand_node, can you explain why a deepcopy of the environment is necessary ?
len(first_level_expansion_nodes)
first_level_expansion_nodes[0].env.render()

frontier.nodes.extend(first_level_expansion_nodes)

# Let's do one step more !

current_node = frontier.nodes.popleft()
current_node.env.render()
current_node.terminated or current_node.truncated

second_level_expansion_nodes = searchAgent.__expand_node__(current_node)
len(second_level_expansion_nodes)
first_level_expansion_nodes[0].env.render()
frontier.nodes.extend(second_level_expansion_nodes)
frontier.nodes[0].env.render()
frontier.nodes[1].env.render()

# Can you perform some extra steps ?

# what about repeating the procedure until done equals True ?
# Let's better let SimpleBaselines do this work for us... :)

