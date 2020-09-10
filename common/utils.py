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

def mini_batch_train(commander1, commander2, nbattalion1, nbattalion2,
                     max_episodes, epsilon_decay_rate, epsilon_init, batch_size):
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
        Regiment1 = Regiment()
        Regiment2 = Regiment()
        RegDeploy1 = UniformIntDeploy(Regiment1, nbattalion1)
        RegDeploy2 = UniformIntDeploy(Regiment2, nbattalion2)
        RegDeploy1.deploy(5, 10, 1, 3)
        RegDeploy2.deploy(5, 10, 1, 3)

        commander1.set_thisRegiment(Regiment1)
        commander1.set_enemyRegiment(Regiment2)
        commander2.set_thisRegiment(Regiment2)
        commander2.set_enemyRegiment(Regiment1)

        Battle = Battlefield(Regiment1, Regiment2)

        state = Battle.state.copy()
        episode_reward = 0
        done = False

        print('*'*80)
        if episode >= 0.2*(max_episodes):
            epsilon = (1. - epsilon_decay_rate) * epsilon
        while not done:
            print('*'*50)
            print('state: ',state)
            print('battalion1_set: ',Regiment1.battalion_set,' battalion2_set: ',Regiment2.battalion_set)
            order1, action1 = commander1.order(state, eps=epsilon)
            order2, action2 = commander2.order(state)
            print('order1: ',order1, ' action1: ',action1)
            print('order2: ',order2, ' action2: ',action2)
            actions.append(action1)

            commander1.deliver_order(order1)
            commander2.deliver_order(order2)
            print('target: ',[Battle.get_regiment1().battalions[i].get_target()
                              for i in range(Battle.get_regiment1().get_full_size())])
            print('target: ',[Battle.get_regiment2().battalions[i].get_target()
                              for i in range(Battle.get_regiment2().get_full_size())])
            Battle.commence_round()

            Battle.update_state()
            next_state = Battle.state.copy()
            print('next state: ',next_state)

            reward, done = Battle.get_reward()
            print('reward: ',reward)
            commander1.replay_buffer.push(state, action1, reward, next_state, done)
            #print('**** replay buffer ****')
            #print(commander1.replay_buffer.buffer)
            #print('**** ************* ****')
            episode_reward += reward

            # start update only when there are at least batch_size number of samples in replay buffer
            if len(commander1.replay_buffer) > batch_size:
                loss = commander1.update(batch_size)

            state = next_state

        episode_rewards.append(episode_reward)
        losses.append(loss)
        print("Episode " + str(episode) + ": " + str(episode_reward), ' loss: ',loss, ' epsilon: ',epsilon)


    return episode_rewards, losses, actions


if __name__ == '__main__':
    from Commander import QCommander, RandomCommander
    import matplotlib.pyplot as plt
    import itertools

    nbattalion1 = 2
    nbattalion2 = 2
    attack1, attackspread1, health1, healthspread1 = 5, 1, 10, 3
    attack2, attackspread2, health2, healthspread2 = 5, 1, 10, 3
    maxbattalion1 = 1
    maxbattalion2 = 1

    regiment1 = Regiment()
    regiment2 = Regiment()
    RegDeploy1 = UniformIntDeploy(regiment1, nbattalion1)
    RegDeploy2 = UniformIntDeploy(regiment2, nbattalion2)
    RegDeploy1.deploy(5, 10, 1, 3)
    RegDeploy2.deploy(5, 10, 1, 3)

    commander1 = QCommander(regiment1, regiment2, maxbattalion1)
    commander2 = RandomCommander(regiment2, regiment1, maxbattalion2)
    commander1.set_order_action_map()
    commander2.set_order_action_map()
    commander1.set_model()

    n_epoch = 5000
    epsilon_decay_rate = 0.01
    epsilon_init = 1
    batch_size = 128
    episode_rewards, losses, actions = mini_batch_train(commander1, commander2, nbattalion1, nbattalion2,
                                                        n_epoch, epsilon_decay_rate, epsilon_init,
                                                        batch_size)

    #torch.save(Commander1.model.state_dict(),'./DQN_model.tar')

    accumulated_reward = np.array(list(itertools.accumulate(episode_rewards)))
    fig = plt.figure()
    plt.plot(accumulated_reward/np.arange(1,n_epoch + 1,1))
    plt.plot(list(range(n_epoch)), [0]*n_epoch,'k--')
    plt.draw()

    fig = plt.figure()
    plt.plot(losses)
    plt.plot(list(range(n_epoch)), [0]*n_epoch,'k--')
    plt.show()

