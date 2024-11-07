from SimpleBaselines.agent.AbstractRLAgent import AbstractRLAgent
from SimpleBaselines.states.State import State
import gymnasium as gym


class Random_RL_Agent(AbstractRLAgent):

    def __init__(self):
        super().__init__()


    def __random_action_decision_function__(self, state: State):
        return state.env.action_space.sample()


    def play(self, environment:gym.Env, max_steps=5000, seed=None):
        # Set action decision function
        # Random agent plays as a random policy sampling from env.action_space.sample()
        observation, info = environment.reset(seed=seed)
        self.initial_state = State(env=environment, observation=observation, info=info)
        self.current_state = None
        self.final_state = None
        self.step = 0
        self.__action_decision_function__ = self.__random_action_decision_function__
        super().__play__(max_steps)
