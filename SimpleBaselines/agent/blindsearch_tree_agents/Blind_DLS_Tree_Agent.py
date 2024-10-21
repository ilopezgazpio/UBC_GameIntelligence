from SimpleBaselines.agent.AbstractTreeGraphAgent import AbstractTreeGraphAgent
from SimpleBaselines.frontier.Node import Node


class Blind_DLS_Tree_Agent(AbstractTreeGraphAgent):

    def __init__(self):
        super().__init__()


    def blind_search(self, first_node: Node, max_steps=5000, max_depth=10):
        # Set frontier ordering function
        # DLS integrates expanded as LIFO
        self.__initialize_frontier__(first_node)
        self.frontier.frontier_insertion_function = self.frontier.__insert_all_left__
        self.frontier.expanded_post_processing_function = self.frontier.__remove_max_depth_nodes__
        self.frontier.max_depth_nodes = max_depth
        super().__blind_search__(max_steps)
