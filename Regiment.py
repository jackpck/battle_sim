from math import floor, ceil

class Brigade:
    def __init__(self, Nregiment, Nsoldier):
        self.__Nregiment = Nregiment
        self.__Nsoldier = Nsoldier # initial number of soldier. Constant
        self.brigade = [] # list of regiments (list of list of soldiers)
        self.regiment_list = [] # list of soldier_list set

    class Regiment:
        def __init__(self):
            self.soldier_list = set() # id of surviving soldier
            self.regiment = [] # list of soldiers (alive or dead)

        class Soldier:
            __slot__ = '__attack', '__health', '__target'
            def __init__(self, attack, health):
                self.__attack = attack
                self.__health = health
                self.__target = None # enemy_id he targets

            def set_target(self, enemy_id):
                self.__target = enemy_id

            def get_target(self):
                return self.__target

            def receive_damage(self, damage):
                self.__health -= damage

            def get_health(self):
                return self.__health

            def get_attack(self):
                return self.__attack

    def get_regiment_size(self):
        return self.__Nsoldier

    def fire(self, enemyBrigade):
        for soldier_id in self.regiment_list[0].soldier_list:
            enemy_id = self.brigade[0][soldier_id].get_target()
            damage = self.brigade[0][soldier_id].get_attack()
            if enemy_id is not None: # an enemy is targeted (i.e. not None)
                enemyBrigade[0][enemy_id].receive_damage(damage)

    def count_KIA(self):
        '''
        If soldier's health <= 0, remove id from the soldier_list
        '''
        temp_soldier_list = self.regiment_list[0].soldier_list.copy()
        for soldier_id in temp_soldier_list:
            if self.brigade[0][soldier_id].get_health() <= 0:
                self.regiment_list[0].soldier_list.remove(soldier_id)

    def reinforcement(self):
        '''
        continuous reinforcement: as soon as a soldier on the frontline is KIA, a reserve will replace him.
        alternative: queue reinforcement: each reservist can only reinforce one specific frontline soldier.
        '''
        if len(self.brigade) != 2:
            raise AssertionError('To call reinforcement, a brigade must have two regiments.')

        while len(self.brigade[0].soldier_list) < self.get_regiment_size() and len(self.brigade[1].soldier_list) > 0:
            reserve_id = self.brigade[1].soldier_list.pop()
            replacement_id = 2*self.get_regiment_size() - len(self.brigade[1].soldier_list)
            self.brigade[0].soldier_list.add(replacement_id) # reservist id starts from Nsoldier + 1
            self.brigade[0][replacement_id] = self.brigade[1][reserve_id]

