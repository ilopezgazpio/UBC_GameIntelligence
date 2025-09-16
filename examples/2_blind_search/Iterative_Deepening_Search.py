#--------------------------------------------
# Iterative Deepening Search Tutorial
# Search Heuristics - University of the Basque Country (UPV/EHU)
# Inigo Lopez-Gazpio
#--------------------------------------------

# Preliminary notes : Make sure you understand the following pre-requisites
# 1.- Python basics and first steps
# 2.- Python openai gym extra environments provided in class
# 3.- Frontier expansion tutorials

# You can now safely keep on with DLS

# Task 1
# First of all analyze the DLS procedure (SimpleBaselines)

#---------------------------
# River Crossing environment
#---------------------------
import gymnasium as gym
from SimpleBaselines.agent.blindsearch_tree_agents.Blind_IDS_Tree_Agent import Blind_IDS_Tree_Agent
from SimpleBaselines.agent.blindsearch_graph_agents.Blind_IDS_Graph_Agent import Blind_IDS_Graph_Agent
from SimpleBaselines.frontier.Node import Node

import pandas as pd
import matplotlib.pyplot as plt

env = gym.make('gym_RiverCrossing:RiverCrossing-v0')
env = env.unwrapped

first_observation = env.reset()
first_node = Node(env, first_observation)
env.render()

# Bling Tree Agent Search
blind_ids_tree_agent = Blind_IDS_Tree_Agent()
blind_ids_tree_agent.blind_search(first_node)

# If solution is found we can explore the solution :)
blind_ids_tree_agent.final_node.depth
blind_ids_tree_agent.final_node.action_history
blind_ids_tree_agent.final_node.observation
blind_ids_tree_agent.final_node.env.render()
# etc

# we can also analyze the search process using the reporting class inside the agent
blind_ids_tree_agent.reporting.log
blind_ids_tree_agent.reporting.plotNumberNodesFrontier()
blind_ids_tree_agent.reporting.plotFrontierMaxDepth()
blind_ids_tree_agent.reporting.plotNodesAddedFrontier(nbins=10)

# Task
# What is happening in this experiment ?

# now lets try another search agent :)
first_observation = env.reset()
first_node = Node(env, first_observation)

# Bling Graph Agent Search
blind_ids_graph_agent = Blind_IDS_Graph_Agent()
blind_ids_graph_agent.blind_search(first_node)

# If solution is found we can explore the solution :)
blind_ids_graph_agent.final_node.depth
blind_ids_graph_agent.final_node.action_history
blind_ids_graph_agent.final_node.observation
blind_ids_graph_agent.final_node.env.render()

# we can also analyze the search process using the reporting class inside the agent
blind_ids_graph_agent.reporting.log
blind_ids_graph_agent.reporting.plotNumberNodesFrontier()
blind_ids_graph_agent.reporting.plotFrontierMaxDepth()
blind_ids_graph_agent.reporting.plotNodesAddedFrontier(nbins=10)



# We can also compute some combined graphs

NumberNodesFrontier = pd.concat(
    [
       blind_ids_tree_agent.reporting.log['Frontier'],
       blind_ids_graph_agent.reporting.log['Frontier']
    ] , axis=1
)
NumberNodesFrontier.columns = ["# Nodes Tree", "# Nodes Graph"]
NumberNodesFrontier.plot(lw=2, marker='*')
plt.show()


MaxDepth = pd.concat(
    [
       blind_ids_tree_agent.reporting.log['F.Max.Depth'],
       blind_ids_graph_agent.reporting.log['F.Max.Depth']
    ] , axis=1
)
MaxDepth.columns = ["Max Depth Tree", "Max Depth Graph"]
MaxDepth.plot(lw=2, marker='*')
plt.show()


Expanded = pd.concat(
    [
       blind_ids_tree_agent.reporting.log['Expanded'].fillna(0),
       blind_ids_graph_agent.reporting.log['Expanded'].fillna(0)
    ] , axis=1
)
Expanded.columns = ["Expanded Tree", "Expanded Graph"]
Expanded.plot(kind="hist", bins=10)
plt.show()

Expanded.plot(kind="kde", lw=2)
plt.show()







#---------------------
# 8-Puzzle environment
#---------------------
env = gym.make('gym_8Puzzle:8Puzzle-v0')
env = env.unwrapped

first_observation = env.reset()
first_node = Node(env, first_observation)

blind_ids_tree_agent = Blind_IDS_Tree_Agent()
blind_ids_tree_agent.blind_search(first_node)


# Let's try graph.search
blind_ids_graph_agent = Blind_IDS_Graph_Agent()
blind_ids_graph_agent.blind_search(first_node)

# Since no repeated states are checked, for GS version, it is "easier" to reach a certain deep. In addition, frontier grows "slower")
# Let's see the point with graphs :)

NumberNodesFrontier = pd.concat(
    [
       blind_ids_tree_agent.reporting.log['Frontier'],
       blind_ids_graph_agent.reporting.log['Frontier']
    ] , axis=1
)
NumberNodesFrontier.columns = ["# Nodes Tree", "# Nodes Graph"]
NumberNodesFrontier.plot(lw=2, marker='*')
plt.show()


MaxDepth = pd.concat(
    [
       blind_ids_tree_agent.reporting.log['F.Max.Depth'],
       blind_ids_graph_agent.reporting.log['F.Max.Depth']
    ] , axis=1
)
MaxDepth.columns = ["Max Depth Tree", "Max Depth Graph"]
MaxDepth.plot(lw=2, marker='*')
plt.show()


Expanded = pd.concat(
    [
       blind_ids_tree_agent.reporting.log['Expanded'],
       blind_ids_graph_agent.reporting.log['Expanded']
    ] , axis=1
)
Expanded.columns = ["Expanded Tree", "Expanded Graph"]
Expanded.plot(kind="hist", bins=10)
plt.show()

Expanded.plot(kind="kde", lw=2)
plt.show()




















'''

TODO

#---------------------
# Sudoku environment
#---------------------


# Problem 3: Sudoku (brief on Formulation)

problem = initialize.problem("data/sudoku-1.txt")
res1 = Breadth.First.Search(problem, count.limit = 1000)   
res2 = Breadth.First.Search(problem, count.limit = 1000, graph.search = T)   
analyze.results(list(res1, res2))
# which deep would we need??

# Conclusion... something more "intelligent" that BFS needs to be used :)
'''
