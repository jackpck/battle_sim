import numpy as np
import random
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

from common.replay_buffers import BasicBuffer
from vanillaDQN.model import DQN

class Commander:
    '''
    Strategy: Choose nbatcommand Battalions in thisRegiment and engage with nbatcommand Battalions in enemyRegiment
    '''
    def __init__(self, thisRegiment, enemyRegiment, nbatcommand):
        '''
        nbatcommand: number of battalion assign to engage in each round.
        '''
        self.nbatcommand = nbatcommand
        self.thisRegiment = thisRegiment
        self.enemyRegiment = enemyRegiment
        self.thisRegiment_size = self.thisRegiment.get_full_size()
        self.enemyRegiment_size = self.enemyRegiment.get_full_size()
        self.order_action_map = {} # mapping order (tuple) to an id (int)
        self.action_order_map = {} # mapping id (int) to an order (tuple)

    def set_thisRegiment(self, thisRegiment):
        self.thisRegiment = thisRegiment
        self.thisRegiment_size = self.thisRegiment.get_full_size()

    def set_enemyRegiment(self, enemyRegiment):
        self.enemyRegiment = enemyRegiment
        self.enemyRegiment_size = self.enemyRegiment.get_full_size()

    def set_nbatcommand(self, nbatcommand):
        self.nbatcommand = nbatcommand

    def set_order_action_map(self):
        k = 0
        # Include None to account for cases when number of surviving battalions < nbatcommand
        if self.nbatcommand == 1:
            for i in [None] + list(range(self.thisRegiment_size)):
                for m in [None] + list(range(self.enemyRegiment_size)):
                    self.order_action_map[(i,m)] = k
                    self.action_order_map[k] = (i,m)
                    k += 1

        elif self.nbatcommand == 2:
            for i in [None] + list(range(self.thisRegiment_size)):
                for j in [None] + list(range(self.thisRegiment_size)):
                    for m in [None] + list(range(self.enemyRegiment_size)):
                        for n in [None] + list(range(self.enemyRegiment_size)):
                            if i != j: # a battalion cannot attack twice in a round.
                                self.order_action_map[(i,j,m,n)] = k
                                self.action_order_map[k] = (i,j,m,n)
                                k += 1

        else:
            raise ValueError('Currently only support nbatcommand = 1 or 2.')

    def order(self, state):
        raise NotImplmentedError('must be implemented by subclass')

    def deliver_order(self, thisorder):
        '''
        thisorder: (order1, order2).
        '''
        self.thisRegiment.offense_set = set(thisorder[:self.nbatcommand])
        for i,bat in enumerate(thisorder[:self.nbatcommand]):
            if bat is not None:
                self.thisRegiment.battalions[bat].set_target(thisorder[i+self.nbatcommand])

    def order_to_action(self, thisorder):
        order_tuple = tuple(i for i in thisorder)
        return self.order_action_map[order_tuple]


class RandomCommander(Commander):
    def __init__(self, thisRegiment, enemyRegiment, nbatcommand):
        super().__init__(thisRegiment, enemyRegiment, nbatcommand)

    def order(self, state):
        # number of valid (surviving) battalions
        n_valid_battalion1 = min(self.nbatcommand, len(self.thisRegiment.battalion_set))
        n_valid_battalion2 = min(self.nbatcommand, len(self.enemyRegiment.battalion_set))
        # this makes sure the order list has the same length regardless of surviving battalions
        offensive_team = [None]*self.nbatcommand
        defensive_team = [None]*self.nbatcommand
        offensive_team[:n_valid_battalion1] = random.sample(self.thisRegiment.battalion_set, n_valid_battalion1)
        defensive_team[:n_valid_battalion2] = random.sample(self.enemyRegiment.battalion_set, n_valid_battalion2)
        thisorder = offensive_team + defensive_team
        return thisorder, self.order_to_action(thisorder)


class QCommander(Commander):
    def __init__(self, thisRegiment, enemyRegiment, nbatcommand,
                 gamma = 0.95,
                 buffer_size = 256):
        super().__init__(thisRegiment, enemyRegiment, nbatcommand)
        self.gamma = gamma
        self.replay_buffer = BasicBuffer(max_size=buffer_size)
        self.device = None
        self.model = None
        self.optimizer = None
        self.MSE_loss = None

    def set_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_dim = self.thisRegiment_size + self.enemyRegiment_size
        output_dim = len(self.order_action_map)
        self.model = DQN(input_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

    def order(self, state, eps=0.2):
        '''
        state: [array of regiment1 health + array of regiment2 health]
        e.g. action: (4, None, None, 4) -> battalion 4 will not attack any enemy battalion (wasteful!). Only happen
        when agent choose according to the q-table (early stage)
        '''
        if np.random.uniform(0,1) < eps:
            n_valid_battalion1 = min(self.nbatcommand, len(self.thisRegiment.battalion_set))
            offensive_team = [None] * self.nbatcommand
            # Can only choose battalions that still survive.
            offensive_team[:n_valid_battalion1] = random.sample(self.thisRegiment.battalion_set, n_valid_battalion1)
            # In theory can attack non-existing battalions. Commander should be able to learn to avoid those order.
            defensive_team = random.sample(list(range(self.enemyRegiment_size)), self.nbatcommand)
            thisorder = offensive_team + defensive_team
            return thisorder, self.order_to_action(thisorder)

        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        #self.model.eval() # need this when forward passing one sample into nn with batchnorm layer
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())

        return self.action_order_map[action], action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = np.array([int(done) for done in dones])
        dones = torch.FloatTensor(dones).to(self.device)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1)) # [batch_size, 1]
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states) # [batch_size, naction]
        max_next_Q = torch.max(next_Q, 1)[0] # [batch_size, 1]
        expected_Q = rewards.squeeze(1) + self.gamma*(1-dones)*max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        #batch =self.replay_buffer.sample_sequence(batch_size)
        batch =self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()



if __name__ == '__main__':
    from Brigade import Regiment
    from Deploy import UniformDeploy
    from Battlefield import Battlefield

    nbattalion1 = 5
    nbattalion2 = 5
    attack1, health1, attack1_spread, health1_spread = 5, 10, 0, 1
    attack2, health2, attack2_spread, health2_spread = 5, 10, 0, 1

    Regiment1 = Regiment()
    Regiment2 = Regiment()
    RegDeploy1 = UniformDeploy(Regiment1, nbattalion1)
    RegDeploy2 = UniformDeploy(Regiment2, nbattalion2)
    RegDeploy1.deploy(attack1, health1, attack1_spread, health1_spread)
    RegDeploy2.deploy(attack2, health2, attack2_spread, health2_spread)

    Battle = Battlefield(Regiment1, Regiment2)

    Commander1 = QCommander(Regiment1, Regiment2, 1)
    Commander2 = RandomCommander(Regiment2, Regiment1, 1)

    Commander1.set_order_action_map()
    Commander2.set_order_action_map()
    Commander1.set_model()

    Regiment1.battalions[0].set_health(-1)
    Regiment1.battalions[1].set_health(-1)
    Regiment1.battalions[2].set_health(-1)
    Regiment1.battalions[3].set_health(-1)
    Regiment1.count_KIA()
    Battle.update_state()
    print(Battle.state)
    order1, action1 = Commander1.order(Battle.state)
    print(order1,' ',action1)
    print(Commander1.order_action_map)

