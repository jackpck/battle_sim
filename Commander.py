import numpy as np
import random
import operator as op
from functools import reduce

import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from common.replay_buffers import BasicBuffer
from vanillaDQN.model import DQN

class Commander:
    '''
    Strategy: Choose maxBattalion Battalions in thisRegiment and engage with maxBattalion Battalions in enemyRegiment
    '''
    def __init__(self, maxBattalion, battlefield):
        self.maxBattalion = maxBattalion
        self.battlefield = battlefield

        self.order_action_map = {} # mapping order (tuple) to an id (int)
        self.action_order_map = {} # mapping id (int) to an order (tuple)

        k = 0
        for i in range(self.battlefield.regiment1_size):
            for j in range(self.battlefield.regiment1_size):
                for m in range(self.battlefield.regiment2_size):
                    for n in range(self.battlefield.regiment2_size):
                        self.order_action_map[(i,j,m,n)] = k
                        self.action_order_map[k] = (i,j,m,n)
                        k += 1

    def order(self, state):
        pass

    def order_to_action(self, thisorder):
        order_tuple = tuple(i for i in thisorder)
        return self.order_action_map[order_tuple]

    def deliver_order(self, thisorder):
        '''
        thisorder is a tuple of size 2*maxBattalion
        '''
        self.battlefield.get_regiment1().offense_set = set(thisorder[:self.maxBattalion])
        for i,bat in enumerate(self.battlefield.get_regiment1().offense_set):
            self.battlefield.get_regiment1().battalions[bat].set_target(thisorder[i+self.maxBattalion])


class RandomCommander(Commander):
    def __init__(self, maxBattalion, battlefield):
        super().__init__(maxBattalion, battlefield)

    def order(self, state):
        offensive_team = random.sample(self.battlefield.get_regiment1().battalion_set, self.maxBattalion)
        defensive_team = random.sample(self.battlefield.get_regiment2().battalion_set, self.maxBattalion)
        thisorder = offensive_team + defensive_team
        return thisorder, self.order_to_action(thisorder)


class QCommander(Commander):
    def __init__(self, maxBattalion, battlefield,
                 learning_rate = 3e-4,
                 gamma = 0.95,
                 buffer_size = 10000):
        super().__init__(maxBattalion, battlefield)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = DQN(battlefield.get_state_size(),len(self.order_action_map)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    @classmethod
    def ncr(self, n,k):
        k = min(k, n-k)
        numer = reduce(op.mul, range(n, n-k, -1), 1)
        denom = reduce(op.mul, range(1, k+1), 1)
        return numer // denom

    def order(self, state, eps=0.2):
        '''
        state: battlefield.state
        '''
        if np.random.uniform(0,1) < eps:
            offensive_team = random.sample(list(range(self.battlefield.regiment1_size)), self.maxBattalion)
            defensive_team = random.sample(list(range(self.battlefield.regiment2_size)), self.maxBattalion)
            thisorder = offensive_team + defensive_team
            return thisorder, self.order_to_action(thisorder)

        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        return self.action_order_map[action], action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch =self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimier.step()



if __name__ == '__main__':
    from Brigade import Regiment, Battalion
    from Deploy import UniformDeploy
    from Battlefield import Battlefield

    Regiment1 = Regiment()
    Regiment2 = Regiment()
    RegDeploy1 = UniformDeploy(Regiment1, 5)
    RegDeploy2 = UniformDeploy(Regiment2, 5)
    RegDeploy1.deploy(5,10,1)
    RegDeploy2.deploy(5,10,1)

    Battle = Battlefield(Regiment1, Regiment2)

    Commander1 = QCommander(2, Battle)
    Commander2 = RandomCommander(2, Battle)

    order1, action1 = Commander1.order(Battle.state)
    print(order1,' ',action1)

    '''
    while Regiment1.battalion_set and Regiment2.battalion_set:
        old_state = Battle.state

        order1, action1 = Commander1.order(Regiment2)
        order2, action2 = Commander2.order(Regiment1)

        Commander1.deliver_order(order1)
        Commander2.deliver_order(order2)
        Battle.commence_round()

        Battle.update_state()
        next_state = Battle.state
        action = action1
        reward = 0
        done = False

        batch = [state, action, reward, next_state, done]

    if Regiment1.battlion_set:
        reward = 1
    else:
        reward = -1
    '''
