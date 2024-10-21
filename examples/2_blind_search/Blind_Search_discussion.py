#---------------------------------------------------------------------------
# Blind search over different problems
# Search Heuristics - University of the Basque Country (UPV/EHU)
# Inigo Lopez-Gazpio
#---------------------------------------------------------------------------

# Preliminary notes : Make sure you understand the following pre-requisites
# 1.- Python basics and first steps
# 2.- Python gymnasium extra environments provided in class
# 3.- Frontier expansion tutorials
# 4.- Analyze individual tutorials for each search method


#---------------------------
# River Crossing environment
#---------------------------
import gymnasium as gym
from SimpleBaselines.frontier.Node import Node

from SimpleBaselines.agent.blindsearch_tree_agents.Blind_BFS_Tree_Agent import Blind_BFS_Tree_Agent
from SimpleBaselines.agent.blindsearch_graph_agents.Blind_BFS_Graph_Agent import Blind_BFS_Graph_Agent
from SimpleBaselines.agent.blindsearch_tree_agents.Blind_DFS_Tree_Agent import Blind_DFS_Tree_Agent
from SimpleBaselines.agent.blindsearch_graph_agents.Blind_DFS_Graph_Agent import Blind_DFS_Graph_Agent
from SimpleBaselines.agent.blindsearch_tree_agents.Blind_DLS_Tree_Agent import Blind_DLS_Tree_Agent
from SimpleBaselines.agent.blindsearch_graph_agents.Blind_DLS_Graph_Agent import Blind_DLS_Graph_Agent
from SimpleBaselines.agent.blindsearch_tree_agents.Blind_IDS_Tree_Agent import Blind_IDS_Tree_Agent
from SimpleBaselines.agent.blindsearch_graph_agents.Blind_IDS_Graph_Agent import Blind_IDS_Graph_Agent
from SimpleBaselines.agent.blindsearch_tree_agents.Blind_UCS_Tree_Agent import Blind_UCS_Tree_Agent
from SimpleBaselines.agent.blindsearch_graph_agents.Blind_UCS_Graph_Agent import Blind_UCS_Graph_Agent

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Environment set up
env = gym.make('gym_RiverCrossing:RiverCrossing-v0')
first_observation = env.reset()
first_node = Node(env, first_observation)

# Agent set up
blind_bfs_tree_agent = Blind_BFS_Tree_Agent()
blind_bfs_graph_agent = Blind_BFS_Graph_Agent()
blind_dfs_tree_agent = Blind_DFS_Tree_Agent()
blind_dfs_graph_agent = Blind_DFS_Graph_Agent()
blind_dls_tree_agent = Blind_DLS_Tree_Agent()
blind_dls_graph_agent = Blind_DLS_Graph_Agent()
blind_ids_tree_agent = Blind_IDS_Tree_Agent()
blind_ids_graph_agent = Blind_IDS_Graph_Agent()
blind_ucs_tree_agent = Blind_UCS_Tree_Agent()
blind_ucs_graph_agent = Blind_UCS_Graph_Agent()

# Search for results
blind_bfs_tree_agent.blind_search(first_node)
blind_bfs_graph_agent.blind_search(first_node)
blind_dfs_tree_agent.blind_search(first_node)
blind_dfs_graph_agent.blind_search(first_node)
blind_dls_tree_agent.blind_search(first_node, max_depth=10)
blind_dls_graph_agent.blind_search(first_node, max_depth=10)
blind_ids_tree_agent.blind_search(first_node, depth_limit=10)
blind_ids_graph_agent.blind_search(first_node, depth_limit=10)
blind_ucs_tree_agent.blind_search(first_node)
blind_ucs_graph_agent.blind_search(first_node)

labels = [
    "blind_bfs_tree_agent",
    "blind_bfs_graph_agent",
    "blind_dfs_tree_agent",
    "blind_dfs_graph_agent",
    "blind_dls_tree_agent",
    "blind_dls_graph_agent",
    "blind_ids_tree_agent",
    "blind_ids_graph_agent",
    "blind_ucs_tree_agent",
    "blind_ucs_graph_agent"
]

agents = [
    blind_bfs_tree_agent,
    blind_bfs_graph_agent,
    blind_dfs_tree_agent,
    blind_dfs_graph_agent,
    blind_dls_tree_agent,
    blind_dls_graph_agent,
    blind_ids_tree_agent,
    blind_ids_graph_agent,
    blind_ucs_tree_agent,
    blind_ucs_graph_agent
]

# Lets do some exploratory analysis
for name, agent in zip(labels, agents):
    print("{}: Solution found {}".format(name, agent.final_node is not None, agent))


for name, agent in zip(labels, agents):
    agent.reporting.log['Frontier'].plot(lw=5, marker='x', markersize=2, title='Iteration vs Frontier nodes', label=name, alpha=0.5)
plt.legend()
plt.xlim(0,500)
plt.ylim(0,50)
plt.show()

# what has happened with DFS tree agent ????
blind_dfs_tree_agent = Blind_DFS_Tree_Agent()
blind_dfs_tree_agent.blind_search(first_node, max_steps=500)
agents[2] = blind_dfs_tree_agent

for name, agent in zip(labels, agents):
    agent.reporting.log['Frontier'].plot(lw=5, marker='x', markersize=2, title='Iteration vs Frontier nodes', label=name, alpha=0.5)
plt.legend()
plt.xlim(0,500)
plt.ylim(0,50)
plt.show()


for name, agent in zip(labels, agents):
    agent.reporting.log['Cur.Depth'].plot(lw=5, marker='x', markersize=2, title='Iteration vs Depth of current expanded', label=name, alpha=0.5)
plt.legend()
plt.xlim(0,50)
plt.ylim(0,50)
plt.show()







#------------------
# 8 puzzle-problem
#------------------

# Let's define the environment
env = gym.make('gym_8Puzzle:8Puzzle-v0')
first_observation = env.reset()

# ... but hack the environment to set an easy setup, instead of random
first_observation = np.array((1,2,5,3,4,8,0,6,7)).reshape(3,3)
# 6 steps needed to be solved (R,R,U,U,L,L)

env.env.state = first_observation
first_node = Node(env, first_observation)

# Check everything OK
env.render()

# For this problem, maximum depth tends to a high number, so DFS can get lost, both using or not graph search
# DLS is effective in this cases, but the limit has to be carefully configured. This is usually very difficult to know

# Agent set up
blind_bfs_tree_agent = Blind_BFS_Tree_Agent()
blind_bfs_graph_agent = Blind_BFS_Graph_Agent()
blind_dfs_tree_agent = Blind_DFS_Tree_Agent()
blind_dfs_graph_agent = Blind_DFS_Graph_Agent()
blind_dls_tree_agent = Blind_DLS_Tree_Agent()
blind_dls_graph_agent = Blind_DLS_Graph_Agent()
blind_ids_tree_agent = Blind_IDS_Tree_Agent()
blind_ids_graph_agent = Blind_IDS_Graph_Agent()
blind_ucs_tree_agent = Blind_UCS_Tree_Agent()
blind_ucs_graph_agent = Blind_UCS_Graph_Agent()

# Search for results
blind_bfs_tree_agent.blind_search(first_node)
blind_bfs_graph_agent.blind_search(first_node)
blind_dfs_tree_agent.blind_search(first_node)
blind_dfs_graph_agent.blind_search(first_node)
blind_dls_tree_agent.blind_search(first_node, max_depth=7)
blind_dls_graph_agent.blind_search(first_node, max_depth=7)
blind_ids_tree_agent.blind_search(first_node, depth_limit=7)
blind_ids_graph_agent.blind_search(first_node, depth_limit=7)
# what happens if limit is incorretly stablished ?
blind_ucs_tree_agent.blind_search(first_node)
blind_ucs_graph_agent.blind_search(first_node)

labels = [
    "blind_bfs_tree_agent",
    "blind_bfs_graph_agent",
    "blind_dfs_tree_agent",
    "blind_dfs_graph_agent",
    "blind_dls_tree_agent",
    "blind_dls_graph_agent",
    "blind_ids_tree_agent",
    "blind_ids_graph_agent",
    "blind_ucs_tree_agent",
    "blind_ucs_graph_agent"
]

agents = [
    blind_bfs_tree_agent,
    blind_bfs_graph_agent,
    blind_dfs_tree_agent,
    blind_dfs_graph_agent,
    blind_dls_tree_agent,
    blind_dls_graph_agent,
    blind_ids_tree_agent,
    blind_ids_graph_agent,
    blind_ucs_tree_agent,
    blind_ucs_graph_agent
]

# Lets do some exploratory analysis
for name, agent in zip(labels, agents):
    print("{}: Solution found {}".format(name, agent.final_node is not None, agent))


for name, agent in zip(labels, agents):
    agent.reporting.log['Frontier'].plot(lw=5, marker='x', markersize=2, title='Iteration vs Frontier nodes', label=name, alpha=0.5)
plt.legend()
#plt.xlim(0,50)
#plt.ylim(0,50)
plt.show()

# what has happened with DFS tree agent ????
blind_dfs_tree_agent = Blind_DFS_Tree_Agent()
blind_dfs_tree_agent.blind_search(first_node, max_steps=500)
agents[2] = blind_dfs_tree_agent

for name, agent in zip(labels, agents):
    agent.reporting.log['Frontier'].plot(lw=5, marker='x', markersize=2, title='Iteration vs Frontier nodes', label=name, alpha=0.5)
plt.legend()
plt.xlim(0,50)
plt.ylim(0,50)
plt.show()


for name, agent in zip(labels, agents):
    agent.reporting.log['Cur.Depth'].plot(lw=5, marker='x', markersize=2, title='Iteration vs Depth of current expanded', label=name, alpha=0.5)
plt.legend()
plt.xlim(0,50)
plt.ylim(0,50)
plt.show()




# Extra experiment. Try changing costs and analyzing performance of UCS with distinct costs.
# We can use OpenAI gym environment wrappers for this. It is a fast way to modify environments, such as to alter costs.
# https://alexandervandekleut.github.io/gym-wrappers/

# Lets remember the action space
env.action_space

# 0 -> Up
# 1 -> Down
# 2 -> Left
# 3 -> Right

class BasicWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        new_cost = np.array((-10,-10,-20,-20))
        # Note we need to convert costs to negative rewards as expected in RL
        new_reward = new_cost[action]
        return next_state, new_reward, done, info

    def __is_applicable__(self, action):
        return self.env.__is_applicable__(action)

altered_env = BasicWrapper(gym.make('gym_8Puzzle:8Puzzle-v0'))
first_observation = altered_env.reset()

# ... but hack the environment to set an easy setup, instead of random
first_observation = np.array((1,2,5,3,4,8,0,6,7)).reshape(3,3)

# 6 steps needed to be solved (R,R,U,U,L,L)
altered_env.env.env.state = first_observation
first_node = Node(altered_env, first_observation)

# Check everything OK
altered_env.render()

altered_blind_ucs_tree_agent = Blind_UCS_Tree_Agent()
altered_blind_ucs_tree_agent.wrapped_env = True
altered_blind_ucs_tree_agent.blind_search(first_node)

altered_blind_ucs_graph_agent = Blind_UCS_Graph_Agent()
altered_blind_ucs_graph_agent.wrapped_env = True
altered_blind_ucs_graph_agent.blind_search(first_node)



MaxDepth = pd.concat(
    [
       blind_ucs_tree_agent.reporting.log['Frontier'],
       blind_ucs_graph_agent.reporting.log['Frontier'],
       altered_blind_ucs_tree_agent.reporting.log['Frontier'],
       altered_blind_ucs_graph_agent.reporting.log['Frontier']
    ] , axis=1
)
MaxDepth.columns = ["Cost 1 tree", "Cost 1 graph", "Cost 2 tree", "Cost 2 graph"]
plt.title("Frontier # nodes per step")
MaxDepth.plot(alpha=0.5, lw=2, marker='.')
plt.xlim(0, 200)
plt.ylim(0, 200)
plt.show()



MaxDepth = pd.concat(
    [
       blind_ucs_tree_agent.reporting.log['Cur.Cost'],
       blind_ucs_graph_agent.reporting.log['Cur.Cost'],
       altered_blind_ucs_tree_agent.reporting.log['Cur.Cost'],
       altered_blind_ucs_graph_agent.reporting.log['Cur.Cost']
    ] , axis=1
)
MaxDepth.columns = ["Cost 1 tree", "Cost 1 graph", "Cost 2 tree", "Cost 2 graph"]
plt.title("Expanded current node cost")
MaxDepth.plot(alpha=0.5, lw=2, marker='.')
plt.show()

# If we change the cost of actions, we are changing the semantics of the search completely,
# if and only if the implementation of the algorithm is aware of costs





'''
TODO
# Let's try with the other problem
source("Sudoku.R")
problem = initialize.problem("data/sudoku-1.txt")
res.BFS = Breadth.First.Search(problem, count.limit = 2500)
res.BFS.gs = Breadth.First.Search(problem, graph.search = T, count.limit = 2500)
res.DFS = Depth.First.Search(problem, count.limit = 2500)
res.DFS.gs = Depth.First.Search(problem, graph.search = T, count.limit = 2500)
analyze.results(list(res.BFS,res.BFS.gs,res.DFS,res.DFS.gs),problem)
# (This problem has "fixed" maximum depth, so DFS can "easily" find the solution)
'''






