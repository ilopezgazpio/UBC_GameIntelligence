class PERDetDQN_RL_Agent(DeterministicDQN_RL_Agent):

    def __init__(self,
                 env:gym.Env,
                 seed=None,
                 gamma=0.99,
                 n_step=1,
                 nn_learning_rate=0.00005,
                 egreedy=0.9,
                 egreedy_final=0.02,
                 egreedy_decay=500,
                 hidden_layers_size=[128, 128],
                 activation_fn=nn.Tanh,
                 dropout=0.0,
                 use_batch_norm=False,
                 loss_fn=nn.MSELoss,
                 optimizer=optim.Adam,
                 memory_size=50000,
                 batch_size=32, 
                 per_alpha=0.3,
                 per_beta=0.4,
                 per_beta_increment=0.0005,
                 per_epsilon=0.01,
                 per_variant='proportional'
                 ):

        super().__init__(env=env,
                         seed=seed,
                         gamma=gamma,
                         nn_learning_rate=nn_learning_rate,
                         egreedy=egreedy,
                         egreedy_final=egreedy_final,
                         egreedy_decay=egreedy_decay,
                         hidden_layers_size=hidden_layers_size,
                         activation_fn=activation_fn,
                         dropout=dropout,
                         use_batch_norm=use_batch_norm,
                         loss_fn=loss_fn,
                         optimizer=optimizer
                         )


        ''' Parameters for the Experience Replay DQN agent '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_step = n_step
        self.n_step_buffer = []
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment
        self.per_epsilon = per_epsilon
        self.per_variant = per_variant

        # Create memory buffer
        self.memory = PrioritizedExperienceReplay(
            memory_size=self.memory_size, 
            batch_size=self.batch_size,
            alpha=self.per_alpha, 
            beta=self.per_beta,
            beta_increment=self.per_beta_increment,
            epsilon=self.per_epsilon,
            variant=self.per_variant
            )

        # Set action decision function
        # Experience Replay Deterministic DQN agent plays as a standard Deterministic DQN
        # internal estimator NN updates following the deterministic scenario Bellman equation under specified loss function

        # action decision function is inherited from superclass
        # self.__action_decision_function__ = self.__DQN_decision_function__

        # we need to update the update function to use experience replay
        self.__update_function__ = self.__PER_DQN_bellman_update__

    def __PER_DQN_bellman_update__(self, old_state: State, action, new_observation : gym.Space, reward: float, terminated, truncated):
        
        exp = old_state, action, new_observation, reward, terminated, truncated # Experiencia/transición actual

        if len(self.n_step_buffer) < self.n_step:
            self.n_step_buffer.append((exp))
            return
        else:  
            # Reward n-step
            R_reward = sum([self.gamma ** i * r for i, (_, _, _, r, _, _) in enumerate(self.n_step_buffer)])
            
            last_old_state, last_action, last_new_observation, last_reward, last_terminated, last_truncated = self.n_step_buffer[-1]
            first_old_state, first_action, first_new_observation, first_reward, first_terminated, first_truncated = self.n_step_buffer.pop(0)

            # Store in memory 
            self.memory.push(first_old_state.observation, first_action, last_new_observation, R_reward, last_terminated, last_truncated)
            
            if (last_terminated or last_truncated):
                self.n_step_buffer.clear()
            else:
                self.n_step_buffer.append(exp) 

        if len(self.memory) < self.batch_size:
            # not enough samples in memory
            return
        
        result = self.memory.sample()
        if result is None:
            return

        (old_observation_batch, action_batch, new_observation_batch, reward_batch, terminated_batch, truncated_batch, weights, indices) = result

        '''DQN Bellman equation update'''
        old_observation_batch = self.QNetwork.toDevice(old_observation_batch)
        action_batch = self.QNetwork.toDevice(action_batch, dType=torch.int64)
        new_observation_batch = self.QNetwork.toDevice(new_observation_batch)
        reward_batch = self.QNetwork.toDevice(reward_batch)
        terminated_batch = self.QNetwork.toDevice(terminated_batch, dType=torch.uint8)
        truncated_batch = self.QNetwork.toDevice(truncated_batch, dType=torch.uint8)
        weights = self.QNetwork.toDevice(weights, dType=torch.float32)

        with torch.no_grad():
            q_values_next = self.QNetwork(new_observation_batch)
            max_q_next = torch.max(q_values_next, 1)[0]
            target = reward_batch + (1 - (terminated_batch | truncated_batch)) * self.gamma ** self.n_step * max_q_next

        q_values_current = self.QNetwork(old_observation_batch)
        current = q_values_current.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        td_errors = torch.abs(target - current).detach().cpu().numpy()

        self.memory.update_priorities(indices, td_errors)

        loss = (weights * (current - target) ** 2).mean()

        self.QNetwork.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.QNetwork.parameters(), max_norm=10.0)
        self.QNetwork.optimizer.step()
