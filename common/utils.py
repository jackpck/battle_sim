import sys

sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

from Battlefield import Battlefield
from Brigade import Regiment
from Deploy import UniformDeploy, UniformIntDeploy

class MiniBatchTrain:
    def __init__(self, nbattalion1, nbattalion2,
                 init_deploy1, init_deploy2):
        self.regiment1 = None
        self.regiment2 = None
        self.nbattalion1 = nbattalion1
        self.nbattalion2 = nbattalion2
        self.init_deploy1 = init_deploy1
        self.init_deploy2 = init_deploy2
        self.commander1 = None
        self.commander2 = None
        self.maxbattalion1 = None
        self.maxbattalion2 = None

    def deploy_regiments(self):
        attack1, health1, attack1_spread, health1_spread = self.init_deploy1
        attack2, health2, attack2_spread, health2_spread = self.init_deploy2
        self.regiment1 = Regiment()
        self.regiment2 = Regiment()
        RegDeploy1 = UniformIntDeploy(self.regiment1, self.nbattalion1)
        RegDeploy2 = UniformIntDeploy(self.regiment2, self.nbattalion2)
        RegDeploy1.deploy(attack1, health1, attack1_spread, health1_spread)
        RegDeploy2.deploy(attack2, health2, attack2_spread, health2_spread)

    def initialize_commanders(self, maxbattalion1, maxbattalion2):
        self.maxbattalion1 = maxbattalion1
        self.maxbattalion2 = maxbattalion2
        self.commander1 = QCommander(self.regiment1, self.regiment2, self.maxbattalion1)
        self.commander2 = RandomCommander(self.regiment2, self.regiment1, self.maxbattalion2)
        self.commander1.set_order_action_map()
        self.commander2.set_order_action_map()
        self.commander1.set_model()

    def mini_batch_train(self, max_episodes, epsilon_decay_rate,
                         epsilon_init, batch_size):
        '''
        commander1: QCommander
        commander2: RandomCommander
        '''
        episode_rewards = []
        actions = []
        losses = []
        loss = -1 # loss of episode with less than batch_size number of samples in replay_buffer
        epsilon = epsilon_init

        for episode in range(max_episodes):
            # initialize battle
            self.deploy_regiments()
            self.commander1.set_thisRegiment(self.regiment1)
            self.commander1.set_enemyRegiment(self.regiment2)
            self.commander2.set_thisRegiment(self.regiment2)
            self.commander2.set_enemyRegiment(self.regiment1)
            Battle = Battlefield(self.regiment1, self.regiment2)

            state = Battle.state.copy()
            episode_reward = 0
            done = False

            # choose random action for the first 20% episodes. Afterwards, choose epilson greedy with epsilon converges
            # to zero.
            if episode >= 0.4*(max_episodes):
                epsilon = (1. - epsilon_decay_rate) * epsilon
            while not done:
                order1, action1 = self.commander1.order(state, eps=epsilon)
                order2, action2 = self.commander2.order(state)
                actions.append(action1)

                self.commander1.deliver_order(order1)
                self.commander2.deliver_order(order2)
                Battle.commence_round()

                Battle.update_state()
                next_state = Battle.state.copy()

                reward, done = Battle.get_reward()
                self.commander1.replay_buffer.push(state, action1, reward, next_state, done)
                episode_reward += reward

                # start update only when there are at least batch_size number of samples in replay buffer
                if len(self.commander1.replay_buffer) > batch_size:
                    loss = self.commander1.update(batch_size)

                state = next_state

            episode_rewards.append(episode_reward)
            losses.append(loss)
            print("Episode " + str(episode) + ": " + str(episode_reward), ' loss: ',loss, ' epsilon: ',epsilon)

        return episode_rewards, losses, actions


if __name__ == '__main__':
    from Commander import QCommander, RandomCommander
    import matplotlib.pyplot as plt

    def moving_average(x, n=20):
        ret = np.cumsum(x, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n-1:]/n

    nbattalion1 = 2
    nbattalion2 = 2
    attack1, health1, attack1_spread, health1_spread = 5, 10, 1, 3
    attack2, health2, attack2_spread, health2_spread = 5, 10, 1, 3
    init_deploy1 = [attack1, health1, attack1_spread, health1_spread]
    init_deploy2 = [attack2, health2, attack2_spread, health2_spread]
    maxbattalion1 = 1
    maxbattalion2 = 1

    DQ_train = MiniBatchTrain(nbattalion1,nbattalion2,
                              init_deploy1, init_deploy2)
    DQ_train.deploy_regiments()
    DQ_train.initialize_commanders(maxbattalion1,maxbattalion2)

    n_epoch = 5000
    epsilon_decay_rate = 0.01
    epsilon_init = 1
    batch_size = 64

    episode_rewards, losses, actions = DQ_train.mini_batch_train(
        n_epoch, epsilon_decay_rate, epsilon_init, batch_size)


    fig = plt.figure()
    plt.plot(moving_average(episode_rewards))
    plt.plot(list(range(n_epoch)), [0]*n_epoch,'k--')
    plt.draw()

    fig = plt.figure()
    plt.plot(losses)
    plt.plot(list(range(n_epoch)), [0]*n_epoch,'k--')
    plt.show()

