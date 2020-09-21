import sys
sys.path.append('../')
from common.utils import DQN_Battle
import matplotlib.pyplot as plt


def moving_average(x, n=20):
    ret = np.cumsum(x, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Define parameters
nbattalion1 = 2
nbattalion2 = 2
attack1, health1, attack1_spread, health1_spread = 5, 10, 1, 3
attack2, health2, attack2_spread, health2_spread = 5, 10, 1, 3
init_deploy1 = [attack1, health1, attack1_spread, health1_spread]
init_deploy2 = [attack2, health2, attack2_spread, health2_spread]
maxbattalion1 = 1
maxbattalion2 = 1

# Setup regiment, battlefield and DQN model (Qcommander)
DQ_train = DQN_Battle(nbattalion1, nbattalion2,
                      init_deploy1, init_deploy2)
DQ_train.deploy_regiments()
DQ_train.initialize_commanders(maxbattalion1, maxbattalion2)

# Define parameters for DQN
n_epoch = 20000
fraction_start_epsilon_greedy = 0.4
epsilon_decay_rate = 0.01
epsilon_init = 1
batch_size = 16

# Train the DQN
episode_rewards, losses, actions = DQ_train.mini_batch_train(
    n_epoch, epsilon_decay_rate,
    fraction_start_epsilon_greedy,
    epsilon_init, batch_size)

DQ_train.save_model()

fig = plt.figure()
plt.plot(moving_average(episode_rewards))
plt.plot(list(range(n_epoch)), [0] * n_epoch, 'k--')
plt.draw()

fig = plt.figure()
plt.plot(losses)
plt.plot(list(range(n_epoch)), [0] * n_epoch, 'k--')
plt.show()

