import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions): #input_shape of the observation environment
        self.mem_size = max_size
        self.mem_cntr = 0 # keep track of the first available space to store in the memory
        self.state_memory1 = np.zeros((self.mem_size, 1600))
        self.state_memory2 = np.zeros((self.mem_size, 25))
        self.new_state_memory1 = np.zeros((self.mem_size, 1600))
        self.new_state_memory2 = np.zeros((self.mem_size, 25))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state1, state2, action, reward, state1_,state2_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory1[index] = np.reshape(state1,1600)
        self.state_memory2[index] = state2
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory1[index] = np.reshape(state1_,1600)
        self.new_state_memory2[index] = state2_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states1 = np.reshape(self.state_memory1[batch],(batch_size,1,40,40))
        # print(np.shape(states1))
        states2 = self.state_memory2[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states1_ = np.reshape(self.new_state_memory1[batch],(batch_size,1,40,40))
        states2_ = self.new_state_memory2[batch]
        dones = self.terminal_memory[batch]

        return states1, states2, actions, rewards, states1_,states2_, dones
