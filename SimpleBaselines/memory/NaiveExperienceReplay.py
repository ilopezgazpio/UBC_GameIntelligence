import random

class NaiveExperienceReplay:

    def __init__(self, memory_size=50000, batch_size=32):

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
        self.position = 0

    def push(self, state, action, new_state, reward, terminated, truncated):

        transition = (state, action, new_state, reward, terminated, truncated)

        if self.position < len(self.memory):
            # Memory is full, overwrite old transitions. Set as default to improve strides
            self.memory[self.position] = transition
        else:
            # Memory is not full yet (may happen at the beginning of training)
            self.memory.append(transition)

        self.position = (self.position + 1) % self.memory_size

    def sample(self):
        # Returns list of states, list of actions, list of new_states, ...
        return zip(*random.sample(self.memory, self.batch_size))

    def __len__(self):
        return len(self.memory)
