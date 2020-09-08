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
    def __init__(self, maxBattalion):
        '''
        maxBattalion: number of battalion assign to engage in each round.
        '''
        self.maxBattalion = maxBattalion
        self.order_action_map = {} # mapping order (tuple) to an id (int)
        self.action_order_map = {} # mapping id (int) to an order (tuple)

    def set_order_action_map(self,regiment1_size, regiment2_size):
        k = 0
        # Include None to account for cases when number of surviving battalions < maxBattalion
        # maxBattalion = 2
        for i in [None] + list(range(regiment1_size)):
            for j in [None] + list(range(regiment1_size)):
                for m in [None] + list(range(regiment2_size)):
                    for n in [None] + list(range(regiment2_size)):
                        if i != j: # a battalion cannot attack twice in a round.
                            self.order_action_map[(i,j,m,n)] = k
                            self.action_order_map[k] = (i,j,m,n)
                            k += 1
        '''
        # maxBattalion = 1
        for i in [None] + list(range(regiment1_size)):
            for m in [None] + list(range(regiment2_size)):
                self.order_action_map[(i,m)] = k
                self.action_order_map[k] = (i,m)
                k += 1
        '''

    def order(self, state, thisRegiment, enemyRegiment):
        raise NotImplmentedError('must be implemented by subclass')

    def deliver_order(self, thisorder, thisRegiment):
        '''
        thisorder: (order1, order2).
        '''
        thisRegiment.offense_set = set(thisorder[:self.maxBattalion])
        for i,bat in enumerate(thisRegiment.offense_set):
            if bat is not None:
                thisRegiment.battalions[bat].set_target(thisorder[i+self.maxBattalion])

    def order_to_action(self, thisorder):
        order_tuple = tuple(i for i in thisorder)
        return self.order_action_map[order_tuple]


class RandomCommander(Commander):
    def __init__(self, maxBattalion):
        super().__init__(maxBattalion)

    def order(self, state, thisRegiment, enemyRegiment):
        # number of valid (surviving) battalions
        n_valid_battalion1 = min(self.maxBattalion, len(thisRegiment.battalion_set))
        n_valid_battalion2 = min(self.maxBattalion, len(enemyRegiment.battalion_set))
        # this makes sure the order list has the same length regardless of surviving battalions
        offensive_team = [None]*self.maxBattalion
        defensive_team = [None]*self.maxBattalion
        offensive_team[:n_valid_battalion1] = random.sample(thisRegiment.battalion_set, n_valid_battalion1)
        defensive_team[:n_valid_battalion2] = random.sample(enemyRegiment.battalion_set, n_valid_battalion2)
        thisorder = offensive_team + defensive_team
        return thisorder, self.order_to_action(thisorder)


class QCommander(Commander):
    def __init__(self, maxBattalion,
                 gamma = 0.95,
                 buffer_size = 256):
        super().__init__(maxBattalion)
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.device = None
        self.model = None
        self.optimizer = None
        self.MSE_loss = None

    def set_model(self, regiment1_size, regiment2_size):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DQN(regiment1_size + regiment2_size, len(self.order_action_map)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def order(self, state, thisRegiment, enemyRegiment, eps=0.2):
        '''
        state: [array of regiment1 health + array of regiment2 health]
        e.g. action: (4, None, None, 4) -> battalion 4 will not attack any enemy battalion (wasteful!). Only happen
        when agent choose according to the q-table (early stage)
        '''
        if np.random.uniform(0,1) < eps:
            n_valid_battalion1 = min(self.maxBattalion, len(thisRegiment.battalion_set))
            offensive_team = [None] * self.maxBattalion
            offensive_team[:n_valid_battalion1] = random.sample(thisRegiment.battalion_set, n_valid_battalion1)
            # In theory can attack non-existing battalions. Commander should be able to learn to avoid those order.
            defensive_team = random.sample(list(range(enemyRegiment.get_full_size())), self.maxBattalion)
            thisorder = offensive_team + defensive_team
            return thisorder, self.order_to_action(thisorder)

        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        self.model.eval() # need this when forward passing one sample into nn with batchnorm layer
        qvals = self.model.forward(state)
        print('**** q-table ****')
        print(qvals.cpu().detach().numpy())
        print('**** ******* ****')
        action = np.argmax(qvals.cpu().detach().numpy())
        #if self.action_order_map[action][1] not in enemyRegiment.battalion_set:
        #    print('suboptimal action')

        return self.action_order_map[action], action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = np.array([int(done) for done in dones])
        dones = torch.FloatTensor(dones).to(self.device)

        print('states.shape: ', states.shape)
        #curr_Q = self.model.forward(states)
        #next_Q = self.model.forward(next_states)
        #max_next_Q = torch.max(next_Q, 1)[0]
        #best_action = torch.argmax(next_Q, 1)[0]
        #expected_Q = curr_Q.clone()
        #expected_Q[:, best_action] = rewards.squeeze(1) + self.gamma*(1-dones)*max_next_Q
        '''
        print('**** states ****')
        print(states)
        print('**** ****** ****')
        print('**** next states ****')
        print(next_states)
        print('**** ****** ****')
        print('**** q-next ****')
        print(next_Q)
        print('**** ********** ****')
        print('**** done ****')
        print(dones)
        print('**** **** ****')
        print('**** q-expected ****')
        print(rewards.squeeze(1) + self.gamma*(1-dones)*max_next_Q)
        print('**** ********** ****')
        '''
        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma*(1-dones)*max_next_Q
        #print('curr_Q: ',curr_Q.shape)
        #print('next_Q: ',next_Q.shape)
        #print('max_next_Q: ',max_next_Q.shape)
        #print('rewards: ',rewards.shape)
        #print('expected_Q: ',expected_Q.shape)

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size, state):
        batch =self.replay_buffer.sample_sequence(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        #qvals = self.model.forward(state)
        #print('**** q-table (updated) ****')
        #print(qvals.cpu().detach().numpy())
        #print('**** ***************** ****')

        return loss.item()



if __name__ == '__main__':
    from Brigade import Regiment
    from Deploy import UniformDeploy
    from Battlefield import Battlefield

    Commander1 = QCommander(1)
    Commander2 = RandomCommander(1)

    Regiment1 = Regiment()
    Regiment2 = Regiment()
    RegDeploy1 = UniformDeploy(Regiment1, 5)
    RegDeploy2 = UniformDeploy(Regiment2, 5)
    RegDeploy1.deploy(5,10,0,1)
    RegDeploy2.deploy(5,10,0,1)

    Battle = Battlefield(Regiment1, Regiment2)
    Commander1.set_order_action_map(5,5)
    Commander2.set_order_action_map(5,5)
    Commander1.set_model(5, 5)

    Regiment1.battalions[0].set_health(-1)
    Regiment1.battalions[1].set_health(-1)
    Regiment1.battalions[2].set_health(-1)
    Regiment1.battalions[3].set_health(-1)
    Regiment1.count_KIA()
    Battle.update_state()
    print(Battle.state)
    order1, action1 = Commander1.order(Battle.state, Regiment1, Regiment2)
    print(order1,' ',action1)
    print(Commander1.order_action_map)

