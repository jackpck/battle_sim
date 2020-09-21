import random
from math import ceil, floor
from Brigade import Regiment, Battalion

class Deploy:
    def __init__(self, thisRegiment, nBattalion):
        self.thisRegiment = thisRegiment
        self.nBattalion = nBattalion

    def deploy(self):
        pass


class DeltaDeploy(Deploy):
    '''
    attack, health follows a delta distribution
    this will create an artifact that odd integer health will favor one team
    while even integer health will favor another.
    '''
    pass



class UniformDeploy(Deploy):
    '''
    attack, health follows a uniform distribution
    a small value of noise will remove the artifact in DeltaDeploy
    '''
    def __init__(self, thisRegiment, nBattalion):
        super().__init__(thisRegiment, nBattalion)

    def deploy(self, attack, health, attack_spread, health_spread):
        for i in range(self.nBattalion):
            self.thisRegiment.battalion_set.add(i)
            self.thisRegiment.battalions.append(Battalion(attack + random.uniform(-attack_spread,attack_spread),
                                                          health + random.uniform(-health_spread,health_spread)))


class UniformIntDeploy(Deploy):
    '''
    attack, health are integers and follows a uniform distribution
    '''
    def __init__(self, thisRegiment, nBattalion):
        super().__init__(thisRegiment, nBattalion)

    def deploy(self, attack, health, attack_spread, health_spread):
        for i in range(self.nBattalion):
            self.thisRegiment.battalion_set.add(i)
            self.thisRegiment.battalions.append(Battalion(attack + random.randint(-attack_spread,attack_spread),
                                                          health + random.randint(-health_spread,health_spread)))


if __name__ == '__main__':
    from Brigade import Regiment, Battalion

    Regiment1 = Regiment()
    RegDeploy = UniformDeploy(Regiment1, 5)
    RegDeploy.deploy(5,10,1)

    print([Regiment1.battalions[i].get_health() for i in Regiment1.battalion_set])
    print([Regiment1.battalions[i].get_attack() for i in Regiment1.battalion_set])



