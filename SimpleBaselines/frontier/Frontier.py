import math
import bisect

from collections import deque
from collections import defaultdict
from SimpleBaselines.frontier.Node import Node



class Frontier:

    def __init__(self):
        self.nodes = deque()
        self.visited_nodes = defaultdict(bool) # default behaviour, node visited = False
        self.frontier_insertion_function = lambda x: x #identity function
        self.frontier_ordering_function = lambda x: x #identity function
        self.expanded_post_processing_function = lambda x: x  # identity function
        self.max_depth_nodes = None


    def __insert_one_left__(self, new_node : Node) -> None:
        self.nodes.appendleft(new_node)


    def __insert_one_right__(self, new_node: Node) -> None:
        self.nodes.append(new_node)


    def __get_one_left__(self) -> Node:
        return self.nodes.popleft()


    def __get_one_right__(self) -> Node:
        return self.nodes.pop()


    def __insert_all_left__(self, list_nodes: list) -> None:
        self.nodes.extendleft(list_nodes)


    def __insert_all_right__(self, list_nodes: list) -> None:
        self.nodes.extend(list_nodes)


    def __insert_all_right_not_visited__(self, list_nodes: list) -> None:
        for node in list_nodes:
            if not self.visited_nodes[tuple(node.observation.reshape(-1).tolist())]:
                self.__insert_one_right__(node)
                self.visited_nodes.update( {tuple(node.observation.reshape(-1).tolist()) : True} )


    def __insert_all_left_not_visited__(self, list_nodes: list) -> None:
        for node in reversed(list_nodes):
            if not self.visited_nodes[tuple(node.observation.reshape(-1).tolist())]:
                self.__insert_one_left__(node)
                self.visited_nodes.update({tuple(node.observation.reshape(-1).tolist()): True})


    def __get_frontier_max_depth__(self):
        return max ( [node.depth for node in self.nodes], default=0 )


    def __remove_max_depth_nodes__(self, list_nodes: list) -> list:
        '''
        This function receives a list of nodes (possibly some expanded nodes on an iteration) and filters the nodes that have depth value > max_depth
        Parameters
        ----------
        list_nodes
        max_depth

        Returns
        -------
        List of nodes
        '''
        return list(filter(lambda x: x.depth <= self.max_depth_nodes, list_nodes))


    def __to_absolute__(self, x: Node) -> Node:
        x.cost = math.abs(x.cost)


    def __absolute_cost_function__(self, expanded : list) -> list:
        '''
        Post-process a node list so that the cost Variable is always positive
        '''
        return map(self.__to_absolute__ , expanded)


    def __insert_ordered_cost__(self, list_nodes: list) -> None:
        '''
        Use this function only if cost is saved as a negative floating value
        As we are using OpenAI reward is negative per step.
        The more negative the worse the node is (more steps)
        '''
        self.nodes.extendleft(list_nodes)
        self.nodes = deque( reversed( sorted( self.nodes, key=lambda x: x.cost)))


    def __insert_ordered_cost_not_visited__(self, list_nodes: list) -> None:
        '''
        Use this function only if cost is saved as a negative floating value
        As we are using OpenAI reward is negative per step.
        The more negative the worse the node is (more steps)
        '''

        for node in list_nodes:
            if not self.visited_nodes[tuple(node.observation.reshape(-1).tolist())]:
                self.__insert_one_left__(node)
                self.visited_nodes.update({tuple(node.observation.reshape(-1).tolist()): True})

        self.nodes = deque( reversed( sorted( self.nodes, key=lambda x: x.cost)))
