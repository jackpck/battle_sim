import random
import numpy as np
from collections import deque

class BasicBuffer:

    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        n_nonzero_rewards = 0
        for experience in batch:
            state, action, reward, next_state, done = experience
            if reward != 0:
                n_nonzero_rewards += 1
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        print('number of non zero rewards in this batch: ',n_nonzero_rewards)
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def sample_sequence(self, batch_size):
        '''
        Sample consecutive samples
        '''
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        min_start = len(self.buffer) - batch_size
        start = np.random.randint(0, min_start)

        n_nonzero_rewards = 0
        for sample in range(start, start + batch_size):
            state, action, reward, next_state, done = self.buffer[sample]
            if reward != 0:
                n_nonzero_rewards += 1
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        #print('number of non zero rewards in this batch: ',n_nonzero_rewards, ' start: ',start)
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)



