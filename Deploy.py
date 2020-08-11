import random

class Deploy:
    def __init__(self):
        pass

    def deploy(self):
        pass


class DeltaDeploy(Deploy):
    '''
    attack, health follows a delta distribution
    this will create an artifact that odd integer health will favor one team
    while even integer health will favor another.
    '''
    def __init__(self, thisBrigade_):
        self.__thisBrigade = thisBrigade_

    def deploy(self, attack, health, reserve=False):
        if reserve:
            self.__thisRegiment.frontline = {i: self.__thisRegiment.Soldier(attack, health)
                                             for i in range(self.__thisRegiment.get_Nsoldier())}
            self.__thisRegiment.soldier_list = set(self.__thisRegiment.frontline.keys())


class UniformDeploy(Deploy):
    '''
    attack, health follows a uniform distribution
    a small value of noise will remove the artifact in DeltaDeploy
    '''
    def __init__(self, thisRegiment_):
        self.__thisRegiment = thisRegiment_

    def deploy(self, attack, health, spread):
        self.__thisRegiment.frontline = {i: self.__thisRegiment.Soldier(attack + random.uniform(-spread,spread),
                                                                        health + random.uniform(-spread,spread))
                                         for i in range(self.__thisRegiment.get_Nsoldier())}
        self.__thisRegiment.soldier_list = set(self.__thisRegiment.frontline.keys())



if __name__ == '__main__':
    from Regiment import Regiment

    team1 = Regiment(5)
    team1_deploy = UniformDeploy(team1)
    team1_deploy.deploy(5,5,1)

    print([team1.frontline[i].get_health() for i in team1.frontline.keys()])
