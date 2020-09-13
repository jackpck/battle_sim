from math import floor, ceil

class Regiment:
    def __init__(self):
        self.battalion_set = set() # id of surviving soldier
        self.offense_set = set()
        self.battalions = [] # list of soldiers (alive or dead)
        self.__target = None

    def get_full_size(self):
        return len(self.battalions)


    def fire(self, enemyRegiment):
        for bat_id in self.offense_set:
            if bat_id is not None and bat_id in self.battalion_set:
                enemy_bat_id = self.battalions[bat_id].get_target()
                damage = self.battalions[bat_id].get_attack()
                if enemy_bat_id is not None: # an enemy is targeted (i.e. not None)
                    enemyRegiment.battalions[enemy_bat_id].receive_damage(damage)

    def count_KIA(self):
        '''
        If soldier's health <= 0, remove id from the soldier_list
        '''
        temp_battalion_set = self.battalion_set.copy()
        for bat_id in temp_battalion_set:
            if self.battalions[bat_id].get_health() <= 0:
                self.battalion_set.remove(bat_id)

    def __del__(self):
        del self.battalion_set
        del self.offense_set
        del self.battalions



class Battalion:
    __slot__ = '__attack', '__health', '__target'
    def __init__(self, attack, health):
        self.__attack = attack
        self.__health = health
        self.__target = None # enemy_id he targets

    def set_target(self, enemy_battalion_id):
        self.__target = enemy_battalion_id

    def get_target(self):
        return self.__target

    def receive_damage(self, damage):
        self.__health -= damage

    def get_health(self):
        return self.__health

    def get_attack(self):
        return self.__attack

    def set_health(self, health):
        self.__health = health




def reinforcement(self):
    # TODO
    pass
    '''
    continuous reinforcement: as soon as a soldier on the frontline is KIA, a reserve will replace him.
    alternative: queue reinforcement: each reservist can only reinforce one specific frontline soldier.
    if len(self.brigade) != 2:
        raise AssertionError('To call reinforcement, a brigade must have two regiments.')

    while len(self.regiment_list[0]) < len(self.brigade[0]) and len(self.regiment_list[1]) > 0:
        reserve_id = self.regiment_list[1].pop()
        replacement_id = len(self.brigade[0]) + len(self.brigade[1]) - len(self.regiment_list[1])
        self.regiment_list[0].add(replacement_id) # reservist id starts from Nsoldier + 1
        self.brigade[0][replacement_id] = self.brigade[1][reserve_id]
    '''

