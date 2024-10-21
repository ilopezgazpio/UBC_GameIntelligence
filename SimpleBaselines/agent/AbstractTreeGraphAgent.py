from abc import ABC, abstractmethod
import gymnasium
from gymnasium import spaces
from copy import deepcopy
from SimpleBaselines.frontier.Node import Node
from SimpleBaselines.frontier.Frontier import Frontier
from SimpleBaselines.reporting.Report import Report

class AbstractTreeGraphAgent(ABC):

    def __init__(self):
        self.step = 0
        self.initial_node = None
        self.final_node = None
        self.frontier = Frontier()
        self.reporting = Report()
        self.wrapped_env = False


    def __initialize_frontier__(self, initial_node : Node):
        self.initial_node = initial_node
        self.frontier.__insert_one_left__(initial_node)


    def __printlog__(self, print_every=100):
        if (self.step % print_every == 0):
            print(self.reporting.log.tail(1))


    def __blind_search__(self, max_steps=5000):
        '''
        This method implements the main search loop and must be implemented with the particularities of each search agent
        '''
        # Dummy node for initialization, terminated or truncated = false
        current = Node()

        while not (current.terminated or current.truncated) and len(self.frontier.nodes) > 0 and self.step <= max_steps:
            current = self.frontier.__get_one_left__()
            expanded = self.__expand_node__(current)

            # Algorithm agnostic implementation on what filter to apply on expanded nodes (such as a depth limit for DLS)
            expanded = self.frontier.expanded_post_processing_function(expanded)

            # Algorithm agnostic implementation on how to insert expanded nodes in frontier
            self.frontier.frontier_insertion_function(expanded)

            self.reporting.__append__(current.terminated or current.truncated, len(expanded),
                                      self.frontier.__get_frontier_max_depth__(), len(self.frontier.nodes),
                                      current.depth, current.cost)
            self.__printlog__(print_every=1)

            self.step += 1

        if current.terminated or current.truncated:
            self.final_node = current


    def __iterative_blind_search__(self, first_node: Node, max_steps=5000, depth_limit=10):
        # Perform an iterative DLS, note that it is important to clear the frontier each time we initiate a new search
        for depth in range(1,depth_limit + 1):
            self.frontier.max_depth_nodes = depth
            self.frontier.nodes.clear()
            self.__initialize_frontier__(first_node)
            self.__blind_search__(max_steps)


    def __save_iteration__(self):
        pass


    def __heuristic__(self, node: Node) -> float:
        '''

        Parameters
        ----------
        node

        Returns
        -------
        Utility value of a node (from the standpoint of the agent)

        '''

        return 0


    def __expand_node__(self, node: Node) -> list:
        '''

        Parameters
        ----------
        node

        Returns
        -------
        a list of Nodes consequence of expanding node

        Note
        -----
        action_space must be an openai gym Space accepted by environment.step(action) for every action in action_space
        Note that not all openai gym environments implement the __is_applicable__ internal method.
        If so __expand_node__ should be overwritten accordingly :)

        '''

        action_space = node.env.action_space
        assert isinstance(action_space, spaces.Space)

        expanded = list()

        # We need this hack to work with Wrapped environments. Do not pay close attention
        if self.wrapped_env:
            __is_applicable__ = node.env.env.env.is_applicable
        else:
            __is_applicable__ = node.env.env.is_applicable

        if isinstance(action_space, spaces.Discrete):
            for action in range(action_space.n):
                if __is_applicable__(action):
                    env = deepcopy(node.env)
                    observation, reward, terminated, truncated, info = env.step(action)
                    new_node = Node( env,
                                     observation,
                                     reward,
                                     terminated,
                                     truncated,
                                     info,
                                     self.__heuristic__(env),
                                     node.cost + reward,
                                     node.action_history + [action],
                                     node.depth + 1
                                     )
                    expanded.append(new_node)
        else:
            raise ("Box / Tuple spaces not yet implemented")

        return expanded
