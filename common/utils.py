import sys

sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

from Battlefield import Battlefield
from Brigade import Regiment
from Deploy import UniformDeploy

def mini_batch_train(commander1, commander2,
                     max_episodes, epsilon_decay_rate, batch_size):
    '''
    commander1: QCommander
    commander2: RandomCommander
    '''

    episode_rewards = []
    losses = []
    loss = -1 # loss of episode with less than batch_size number of samples in replay_buffer
    epsilon = 1

    for episode in range(max_episodes):
        # initialize battle
        Regiment1 = Regiment()
        Regiment2 = Regiment()
        RegDeploy1 = UniformDeploy(Regiment1, 5)
        RegDeploy2 = UniformDeploy(Regiment2, 5)
        RegDeploy1.deploy(5, 10, 0, 3)
        RegDeploy2.deploy(5, 10, 0, 3)

        Battle = Battlefield(Regiment1, Regiment2)

        state = Battle.state
        episode_reward = 0
        done = False

        print('*'*80)
        epsilon = (1. - epsilon_decay_rate) * epsilon
        while not done:
            print('*'*50)
            print('state: ',state)
            order1, action1 = commander1.order(state, Regiment1, Regiment2, eps=epsilon)
            order2, action2 = commander2.order(state, Regiment2, Regiment1)
            print('order1: ',order1, ' action1: ',action1)

            commander1.deliver_order(order1, Regiment1)
            commander2.deliver_order(order2, Regiment2)
            print('target: ',[Regiment1.battalions[i].get_target() for i in range(Regiment1.get_full_size())])
            Battle.commence_round()

            Battle.update_state()
            next_state = Battle.state

            reward, done = Battle.get_reward()
            print('reward: ',reward)
            commander1.replay_buffer.push(state, action1, reward, next_state, done)
            episode_reward += reward

            if len(commander1.replay_buffer) > batch_size: # start update only when there are at least batch_size
                                                           # number of samples in replay_buffer.
                loss = commander1.update(batch_size)

            state = next_state

        episode_rewards.append(episode_reward)
        losses.append(loss)
        print("Episode " + str(episode) + ": " + str(episode_reward), ' loss: ',loss, ' epsilon: ',epsilon)


    return episode_rewards, losses


if __name__ == '__main__':
    from Commander import QCommander, RandomCommander
    import matplotlib.pyplot as plt
    import itertools

    Commander1 = QCommander(1)
    Commander2 = RandomCommander(1)

    Commander1.set_order_action_map(5,5)
    Commander2.set_order_action_map(5,5)
    Commander1.set_model(5, 5)

    n_epoch = 1000
    epsilon_decay_rate = 0.005
    episode_rewards, losses = mini_batch_train(Commander1, Commander2,
                                               n_epoch, epsilon_decay_rate, 128)

    torch.save(Commander1.model.state_dict(),'./DQN_model.tar')

    accumulated_reward = np.array(list(itertools.accumulate(episode_rewards)))
    fig = plt.figure()
    plt.plot(accumulated_reward/np.arange(1,n_epoch + 1,1))
    plt.plot(list(range(n_epoch)), [0]*n_epoch,'k--')
    plt.draw()

    fig = plt.figure()
    plt.plot(losses)
    plt.plot(list(range(n_epoch)), [0]*n_epoch,'k--')

    plt.show()


