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
                     max_episodes, batch_size):
    '''
    commander1: QCommander
    commander2: RandomCommander
    '''

    episode_rewards = []

    for episode in range(max_episodes):
        # initialize battle
        Regiment1 = Regiment()
        Regiment2 = Regiment()
        RegDeploy1 = UniformDeploy(Regiment1, 5)
        RegDeploy2 = UniformDeploy(Regiment2, 5)
        RegDeploy1.deploy(5, 10, 1)
        RegDeploy2.deploy(5, 10, 1)

        Battle = Battlefield(Regiment1, Regiment2)

        state = Battle.state
        episode_reward = 0
        done = False

        while not done:
            print('*'*50)
            order1, action1 = commander1.order(state, Regiment1, Regiment2)
            order2, action2 = commander2.order(state, Regiment2, Regiment1)

            commander1.deliver_order(order1, Regiment1)
            commander2.deliver_order(order2, Regiment2)
            Battle.commence_round()

            Battle.update_state()
            next_state = Battle.state

            print('next state: ',next_state)
            reward = Battle.get_reward()
            print('reward: ',reward)
            done = reward is not False # win, lose, draw -> True, else -> False
            commander1.replay_buffer.push(state, action1, reward, next_state, done)
            episode_reward += reward

            if len(commander1.replay_buffer) > batch_size:
                commander1.update(batch_size)

            state = next_state

        episode_rewards.append(episode_reward)
        print("Episode " + str(episode) + ": " + str(episode_reward))


    return episode_rewards


if __name__ == '__main__':
    from Commander import QCommander, RandomCommander
    import matplotlib.pyplot as plt
    import itertools

    Commander1 = QCommander(2)
    Commander2 = RandomCommander(2)

    Commander1.set_order_action_map(5,5)
    Commander2.set_order_action_map(5,5)
    Commander1.set_model(5, 5)

    episode_rewards = mini_batch_train(Commander1, Commander2, 1000, 16)

    accumulated_reward = np.array(list(itertools.accumulate(episode_rewards)))
    plt.plot(accumulated_reward/np.arange(1,1001,1))
    plt.plot(list(range(1000)), [0]*1000,'k--')
    plt.show()


