'''
Rule of engagement
'''

import numpy as np

class ROE:
    def __init__(self):
        pass

    def aim(self, enemyBrigade_):
        pass

class fully_connected_ROE(ROE):
    '''
    fully connected ROE
    '''

    def __init__(self, thisBrigade_):
        self.__thisBrigade = thisBrigade_

    def aim(self, enemyBrigade):
        enemy_soldier_list = list(enemyBrigade.regiment_list[0].soldier_list)
        for soldier_id in self.__thisBrigade.regiment_list[0].soldier_list:
            enemy_targeted = np.random.choice(enemy_soldier_list)
            self.__thisBrigade.brigade[0][soldier_id].set_target(enemy_targeted)

class range_ROE(ROE):
    '''
    finite range ROE
    '''

    def __init__(self, thisBrigade, r):
        self.__thisBrigade = thisBrigade
        self.__range = r

    def aim(self, enemyRegiment):
        pass

