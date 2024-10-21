from SimpleBaselines.agent.AbstractTreeGraphAgent import AbstractTreeGraphAgent
from SimpleBaselines.frontier.Node import Node


class Blind_IDS_Graph_Agent(AbstractTreeGraphAgent):

    def __init__(self):
        super().__init__()


    def blind_search(self, first_node:Node, max_steps=20000, depth_limit=10):
        # Set frontier ordering function
        # IDS integrates expanded as LIFO
        # Graph search needs to remember visited nodes before inserting extended nodes to frontier
        self.frontier.frontier_insertion_function = self.frontier.__insert_all_left_not_visited__
        self.frontier.expanded_post_processing_function = self.frontier.__remove_max_depth_nodes__
        self.__iterative_blind_search__(first_node, max_steps, depth_limit)
