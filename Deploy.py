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
        regiment = self.__thisBrigade.Regiment()
        self.__thisBrigade.brigade.append({i: regiment.Soldier(attack, health)
                                             for i in range(self.__thisBrigade.get_brigade_size())})
        self.__thisBrigade.regiment_list.append(set(self.__thisBrigade.brigade[0].keys()))
        if reserve:
            regiment = self.__thisBrigade.Regiment()
            self.__thisBrigade.brigade.append({i: regiment.Soldier(attack, health)
                                               for i in range(self.__thisBrigade.get_brigade_size())})
            self.__thisBrigade.regiment_list.append(set(self.__thisBrigade.brigade[1].keys()))


class UniformDeploy(Deploy):
    '''
    attack, health follows a uniform distribution
    a small value of noise will remove the artifact in DeltaDeploy
    '''
    def __init__(self, thisBrigade_):
        self.__thisBrigade = thisBrigade_

    def deploy(self, attack, health, spread, reserve=False):
        regiment = self.__thisBrigade.Regiment()
        self.__thisBrigade.brigade.append({i: regiment.Soldier(attack + random.uniform(-spread,spread),
                                                               health + random.uniform(-spread,spread))
                                           for i in range(self.__thisBrigade.get_regiment_size())})
        self.__thisBrigade.regiment_list.append(set(self.__thisBrigade.brigade[0].keys()))
        if reserve:
            regiment = self.__thisBrigade.Regiment()
            self.__thisBrigade.brigade.append({i: regiment.Soldier(attack + random.uniform(-spread,spread),
                                                                   health + random.uniform(-spread,spread))
                                               for i in range(self.__thisBrigade.get_regiment_size())})
            self.__thisBrigade.regiment_list.append(set(self.__thisBrigade.brigade[1].keys()))


if __name__ == '__main__':
    from Brigade import Brigade

    team1 = Brigade(5)
    team1_deploy = UniformDeploy(team1)
    team1_deploy.deploy(5, 5 ,1 ,reserve=True)

    print([team1.brigade[0][i].get_attack() for i in range(5)])
    print([team1.brigade[1][i].get_attack() for i in range(5)])
