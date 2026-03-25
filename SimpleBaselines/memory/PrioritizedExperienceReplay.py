class PrioritizedExperienceReplay:

    def __init__(self, memory_size=50000, batch_size=32, 
                 alpha=0.3,
                 beta=0.4, 
                 beta_increment=0.0005, 
                 epsilon=0.01,
                 variant='proportional'): 
        """
        Implementation of Prioritized Experience Replay
        Supports two variants:
            - proportional: pi = |di| + epsilon
            - rank-based: pi = 1 / rank(i)
        """

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.variant = variant

        self.memory = [] 
        self.position = 0
        self.priorities = np.zeros(memory_size, dtype=np.float32) 
        self.size = 0

        if variant == 'rank-based':
            self.num_segments = batch_size 
            self.segment_boundaries = None 
            self._update_segment_boundaries()

    # Stratified Rank-based sampling
    def _update_segment_boundaries(self):
        """
        Update the segment boundaries for rank-based sampling
        """
        if self.size == 0:
            return
        
        priorities = np.array([(1.0 / (i + 1)) ** self.alpha for i in range(self.size)])
        total_prob = priorities.sum()
        prob_acum = np.cumsum(priorities)
        segment_prob = total_prob / self.num_segments 
        
        boundaries = [0]
        target = segment_prob

        for i in range(self.size):
            if prob_acum[i] >= target:
                boundaries.append(i)
                target += segment_prob

        if len(boundaries) < self.num_segments + 1:
            boundaries.append(self.size - 1)

        self.segment_boundaries = boundaries

    def push(self, state, action, new_state, reward, terminated, truncated):
        """
        Add a transition to the replay buffer
        """
        transition = (state, action, new_state, reward, terminated, truncated)
        max_priority = np.max(self.priorities[:self.size]) if self.size > 0 else 1.0

        if self.position < len(self.memory):
            # Memory is full, overwrite
            self.memory[self.position] = transition
            self.priorities[self.position] = max_priority
        else:
            # Memory is not full yet (may happen at the beginning of training)
            self.memory.append(transition)
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)
        
    def _get_probabilities(self, indices):
        """
        Sampling probability: P(i) = pi ** alpha / SUM(pk ** alpha)
        """
        if self.size == 0:
            return np.ones(len(indices)) / len(indices)
        
        if self.variant == 'proportional':
            priorities = self.priorities[indices] ** self.alpha 
            sum_priorities = np.sum(self.priorities[:self.size] ** self.alpha)
        else: 
            ranks = np.arange(len(indices)) + 1
            priorities = (1.0 / ranks) ** self.alpha
            sum_priorities = np.sum([(1.0 / (i + 1)) ** self.alpha for i in range(self.size)])
        
        probabilities = priorities / sum_priorities
        return probabilities
    
    def _get_importance_weights(self, indices, probabilities):
        """
        Weights for importance sampling
        wi = (1/N * 1/Pi) ** beta
        wi = wi / max(wi)
        """
        if self.size == 0:
            return np.ones(len(indices))
        
        N = self.size
        weights = (1.0 / (N * probabilities + 1e-10)) ** self.beta 
        weights = weights / np.max(weights)

        return weights

    def sample(self):
        """
        It selectively chooses which experiences to use for training
        """
        if self.size < self.batch_size:
            return None
        
        if self.variant == 'proportional':
            probabilities = self.priorities[:self.size] ** self.alpha + self.epsilon
            probabilities = probabilities / np.sum(probabilities)

            indices = np.random.choice(self.size, self.batch_size, p=probabilities)
        else: # rank-based
            if self.segment_boundaries is None or len(self.segment_boundaries) <= self.batch_size:
                self._update_segment_boundaries()
            
            indices = []
            for i in range(self.batch_size):
                start = self.segment_boundaries[i]
                end = self.segment_boundaries[i+1] 
                
                if end > start:
                    idx = np.random.randint(start, min(end, self.size))
                else:
                    idx = start
                indices.append(idx)
            indices = np.array(indices) 

        probabilities = self._get_probabilities(indices)

        # Bias correction
        weights = self._get_importance_weights(indices, probabilities)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        transitions = [self.memory[idx] for idx in indices]
        states, actions, new_states, rewards, terminateds, truncateds = zip(*transitions)

        return (list(states), list(actions), list(new_states), list(rewards), list(terminateds), list(truncateds),
                weights, indices) 
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities based on new TD errors
        """
        for idx, td_error in zip(indices, td_errors):
            if idx < self.size:
                clipped = min(abs(float(td_error)), 10.0)
                self.priorities[idx] = clipped + self.epsilon

        if self.variant == "rank-based":
            sort_idx = np.argsort(-self.priorities[:self.size]) 
            self.priorities[:self.size] = self.priorities[sort_idx]

            self.memory = [self.memory[i] for i in sort_idx] 
            

    def __len__(self):
        return self.size
