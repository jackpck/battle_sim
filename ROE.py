'''
Rule of engagement
'''

import numpy as np

class ROE:
    def __init__(self):
        pass

    def aim(self, thisRegiment_, enemyBrigade_):
        pass

class fully_connected_ROE(ROE):
    '''
    fully connected ROE
    '''

    def aim(self, thisRegiment, enemyRegiment):
        enemy_soldier_set = list(enemyRegiment.soldier_set)
        for soldier_id in thisRegiment.soldier_set:
            enemy_targeted = np.random.choice(enemy_soldier_set)
            thisRegiment.regiment[soldier_id].set_target(enemy_targeted)

class range_ROE(ROE):
    '''
    finite range ROE
    '''

    def __init__(self, r):
        self.__range = r

    def aim(self, thisRegiment, enemyRegiment):
        pass

