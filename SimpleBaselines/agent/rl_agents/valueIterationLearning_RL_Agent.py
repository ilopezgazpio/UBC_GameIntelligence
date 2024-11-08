from SimpleBaselines.agent.AbstractRLAgent import AbstractRLAgent
from SimpleBaselines.states.State import State
import gymnasium as gym
import torch


class ValueIterationLearning_RL_Agent(AbstractRLAgent):

    def __init__(self, env:gym.Env, seed=None, gamma=0.9):
        super().__init__(env)
        self.reset_env(seed=seed)

        '''Parameters for the Value Iteration Learning algorithm'''
        self.V_init_steps = 2000
        # V values table - Value for each state ( Value function or V(S) ), collects value for each of the states
        self.V = torch.zeros( [self.env.observation_space.n] )
        self.gamma = gamma
        # Build policy table
        self.policy = torch.zeros( self.env.observation_space.n )

        # Set action decision function
        # Value Iteration agent plays as a V-Based policy from off-policy learning
        # V table updates following the deterministic off-policy equation
        self.__action_decision_function__ = self.__ValueIterationLearning_decision_function__

        # Value iteration does not use an online update function
        # self.__update_function__ = will use default pass function



    def __ValueIterationLearning_init_V_table_init_policy__(self):
        ''' Function to build V values from scratch visiting all possible states '''
        # this is value based on experiments, after that many iterations values don't change significantly any more
        # Build V table
        self.V = torch.zeros(self.env.observation_space.n)

        for _ in range(self.V_init_steps):

            for state in range( self.env.observation_space.n ):
                current = State()
                current.observation = state
                max_value, _ = self.__ValueIterationLearning_best_move__(current)
                self.V[state] = max_value.item()

        for state in range( self.env.observation_space.n ):
            current = State()
            current.observation = state
            _, index = self.__ValueIterationLearning_best_move__(current)
            self.policy[state] = index.item()



    def __ValueIterationLearning_best_move__(self, old_state: State):
        ''' Function to return best possible move out of total possible moves given a state and the V values'''
        actions_V = torch.zeros( self.env.action_space.n )

        for action_possible in range( self.env.action_space.n ):

            # Explore hypothetical action
            for prob, new_state, reward, _ in self.env.env.P[old_state.observation][action_possible]:
                actions_V[action_possible] += (prob * (reward + self.gamma * self.V[new_state]))

        max_value, index = torch.max(actions_V, 0)

        return max_value, index


    def __ValueIterationLearning_decision_function__(self, old_state: State):
        ''' Value Iteration Learning decision function '''
        action = self.policy[old_state.observation]
        return action.item()


    def play(self, max_steps=5000, seed=None):
        self.step = 0
        super().__play__(max_steps)
