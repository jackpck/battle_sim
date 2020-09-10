import numpy as np



class Battlefield:
    def __init__(self, regiment1, regiment2):
        self.__regiment1 = regiment1
        self.__regiment2 = regiment2
        self.regiment1_size = len(self.__regiment1.battalions)
        self.regiment2_size = len(self.__regiment2.battalions)
        self.state = np.array(
            [self.__regiment1.battalions[i].get_health() for i in range(self.regiment1_size)]+
            [self.__regiment2.battalions[i].get_health() for i in range(self.regiment2_size)])

    def get_regiment1(self):
        return self.__regiment1

    def get_regiment2(self):
        return self.__regiment2

    def get_state_size(self):
        return len(self.state)

    def update_state(self):
        for i in range(self.regiment1_size):
            self.state[i] = self.__regiment1.battalions[i].get_health()
        for i in range(self.regiment2_size):
            self.state[i + self.__regiment1.get_full_size()] = self.__regiment2.battalions[i].get_health()

    def commence_round(self):
        self.__regiment1.fire(self.__regiment2)
        self.__regiment2.fire(self.__regiment1)

        self.__regiment1.count_KIA()
        self.__regiment2.count_KIA()

    def commence(self):
        while self.__regiment1.battalion_set and self.__regiment2.battalion_set:
            self.commence_round()

    def get_reward(self):
        if not self.__regiment1.battalion_set or not self.__regiment2.battalion_set:
            result = len(self.__regiment1.battalion_set) - len(self.__regiment2.battalion_set)
            if result > 0:
                return 1, True # regiment1 win, done
            elif result < 0:
                return -1, True # regiment2 win, done
            else:
                return 0, True # draw, done
        else:
            return 0, False # no reward. game not done.


if __name__ == '__main__':

    pass

