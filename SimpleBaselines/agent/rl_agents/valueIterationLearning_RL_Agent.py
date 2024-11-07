from SimpleBaselines.agent.AbstractRLAgent import AbstractRLAgent
from SimpleBaselines.states.State import State
import gymnasium as gym
import torch


class ValueIterationLearning_RL_Agent(AbstractRLAgent):

    def __init__(self, env:gym.Env, gamma=0.9):
        super().__init__()
        self.number_states = env.observation_space.n
        self.number_actions = env.action_space.n

        # V values table - Value for each state ( Value function or V(S) ), collects value for each of the states
        self.V = torch.zeros( [self.number_states] )
        self.gamma = gamma


    def __ValueIterationLearning_init_V_table_init_policy__(self):
        ''' Function to build V values from scratch visiting all possible states '''
        # this is value based on experiments, after that many iterations values don't change significantly any more
        # Build V table
        self.V = torch.zeros(self.number_states)

        for _ in range(self.V_init_steps):

            for state in range( self.number_states ):
                current = State()
                current.observation = state
                current.env = self.initial_state.env

                max_value, _ = self.__ValueIterationLearning_best_move__(current)
                self.V[state] = max_value.item()

        # Build policy table
        self.policy = torch.zeros( self.number_states )

        for state in range( self.number_states ):
            current = State()
            current.observation = state
            current.env = self.initial_state.env

            _, index = self.__ValueIterationLearning_best_move__(current)
            self.policy[state] = index.item()


    def __ValueIterationLearning_best_move__(self, old_state: State):
        ''' Function to return best possible move out of total possible moves given a state and the V values'''
        actions_V = torch.zeros( self.number_actions )

        for action_possible in range( self.number_actions ):

            # Explore hypothetical action
            for prob, new_state, reward, _ in old_state.env.env.P[old_state.observation][action_possible]:
                actions_V[action_possible] += (prob * (reward + self.gamma * self.V[new_state]))

        max_value, index = torch.max(actions_V, 0)

        return max_value, index

    def __ValueIterationLearning_decision_function__(self, old_state: State):
        ''' Value Iteration Learning decision function '''
        action = self.policy[old_state.observation]
        return action.item()

    def play(self, environment:gym.Env, max_steps=5000, seed=None):
        # Set action decision function
        # Value Iteration agent plays as a V-Based policy from off-policy learning
        # V table updates following the deterministic off-policy equation
        observation, info = environment.reset(seed=seed)
        self.initial_state = State(env=environment, observation=observation, info=info)
        self.current_state = None
        self.final_state = None
        self.step = 0
        self.__action_decision_function__ = self.__ValueIterationLearning_decision_function__
        # Value iteration does not use an update function
        #self.__update_function__ = Will use default pass function
        super().__play__(max_steps)
