import random
from SimpleBaselines.memory.NaiveExperienceReplay import NaiveExperienceReplay

class MixedExperienceReplay(NaiveExperienceReplay):
    
    def __init__(self, memory_size=50000, data_size=50000, batch_size=32, data_amount_start=1.0, data_amount_end=0.0, data_amount_decay=0.99):
        super().__init__(memory_size=memory_size, batch_size=batch_size)
        self.data_amount = data_amount_start
        self.data_amount_end = data_amount_end
        self.data_amount_decay = data_amount_decay
        self.data_size = data_size
        self.data = []
        self.data_position = 0

    def __push_to_data__(self, state, action, new_state, reward, terminated, truncated):
        transition = (state, action, new_state, reward, terminated, truncated)

        if self.data_position < len(self.data):
            self.data[self.data_position] = transition
        else:
            self.data.append(transition)

        self.data_position = (self.data_position + 1) % self.data_size

    def populate_data(self, data: list):
        assert len(data[0]) == 6
        for transition in data:
            self.__push_to_data__(*transition)

    def sample(self):
        """
        Sample from data and memory
        :param data_amount: float, percentage of data to sample [0, 1]
        :return: list of tuples (state, action, next_state, reward, terminated, truncated)
        """
        # Returns list of states, list of actions, list of new_states, ...
        data_size = int(self.data_amount * self.batch_size)
        self.data_amount = max(self.data_amount_end, self.data_amount * self.data_amount_decay)
        experience_size = int(self.batch_size - data_size)
        data_sample = random.sample(self.data, data_size)
        experience_sample = random.sample(self.memory, experience_size)
        assert len(data_sample) == data_size
        assert len(experience_sample) == experience_size
        state, action, next_state, reward, terminated, truncated = zip(*data_sample, *experience_sample)
        return zip(*data_sample, *experience_sample)
            