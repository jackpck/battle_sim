import sys

sys.path.append('../')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

from Battlefield import Battlefield
from Brigade import Regiment
from Deploy import UniformDeploy, UniformIntDeploy
from utils import MiniBatchTrain


nbattalion1 = 2
nbattalion2 = 2
attack1, health1, attack1_spread, health1_spread = 5, 10, 1, 3
attack2, health2, attack2_spread, health2_spread = 5, 10, 1, 3
init_deploy1 = [attack1, health1, attack1_spread, health1_spread]
init_deploy2 = [attack2, health2, attack2_spread, health2_spread]
maxbattalion1 = 1
maxbattalion2 = 1

DQ_test = MiniBatchTrain(nbattalion1, nbattalion2,
                          init_deploy1, init_deploy2)

DQ_test.deploy_regiments()
DQ_test.initialize_commanders(maxbattalion1, maxbattalion2)

DQ_test.commander1.model.load_state_dict(torch.load('DQN_battle.tar'))

regiment1_health = [5,5]
action = 0
Q = DQ_test.visualize_Q(regiment1_health,6)
plt.imshow(Q, cmap='hot')
plt.colorbar()
plt.show()
