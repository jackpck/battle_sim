'''
Rule of engagement
'''

import numpy as np
from Regiment import Regiment
from math import floor, ceil

class ROE:
    def __init__(self):
        pass

    def aim(self, enemyRegiment):
        pass

class fully_connected_ROE(ROE):
    '''
    fully connected ROE
    '''

    def __init__(self, thisRegiment):
        self.__thisRegiment = thisRegiment

    def aim(self, enemyRegiment):
        enemy_soldier_list = list(enemyRegiment.soldier_list)
        for soldier_id in self.__thisRegiment.soldier_list:
            enemy_targeted = np.random.choice(enemy_soldier_list)
            self.__thisRegiment.frontline[soldier_id].set_target(enemy_targeted)

class range_ROE(ROE):
    '''
    finite range ROE
    '''

    def __init__(self, thisRegiment, r):
        self.__thisRegiment = thisRegiment
        self.__range = r

    def aim(self, enemyRegiment):
        enemy_soldier_list = list(enemyRegiment.soldier_list)
        for soldier_id in self.__thisRegiment.soldier_list:

