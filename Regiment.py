from math import floor, ceil

class Regiment:
    def __init__(self, Nsoldier):
        self.__Nsoldier = Nsoldier
        self.soldier_list = set() # id of surviving soldier
        self.frontline = [] # list of soldiers (alive or dead)

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

    def get_Nsoldier(self):
        return self.__Nsoldier

    def deploy(self, attack, health):
        '''
        setup the frontline and setip the attack and health of each soldier
        '''
        self.frontline = {i:self.Soldier(attack, health) for i in range(self.__Nsoldier)}
        self.soldier_list = set(self.frontline.keys())

    def fire(self, enemyRegiment):
        for soldier_id in self.soldier_list:
            enemy_id = self.frontline[soldier_id].get_target()
            damage = self.frontline[soldier_id].get_attack()
            if enemy_id is not None: # an enemy is targeted (i.e. not None)
                enemyRegiment.frontline[enemy_id].receive_damage(damage)

    def count_KIA(self):
        '''
        If soldier's health <= 0, remove id from the soldier_list
        '''
        temp_soldier_list = self.soldier_list.copy()
        for soldier_id in temp_soldier_list:
            if self.frontline[soldier_id].get_health() <= 0:
                self.soldier_list.remove(soldier_id)
