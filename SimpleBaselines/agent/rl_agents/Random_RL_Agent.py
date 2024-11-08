from SimpleBaselines.agent.AbstractRLAgent import AbstractRLAgent
from SimpleBaselines.states.State import State
import gymnasium as gym


class Random_RL_Agent(AbstractRLAgent):

    def __init__(self, env:gym.Env, seed=None):
        super().__init__(env)

        observation, info = self.env.reset(seed=seed)
        self.initial_state = State(observation=observation, info=info)
        self.current_state = self.initial_state
        self.final_state = None
        self.step = 0

        # Set action decision function
        # Random agent plays as a random policy sampling from env.action_space.sample()
        self.__action_decision_function__ = self.__random_action_decision_function__


    def __random_action_decision_function__(self, state: State):
        return self.env.action_space.sample()


    def play(self, max_steps=5000, seed=None):
        super().__play__(max_steps, seed=seed)
