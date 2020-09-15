from ROE import fully_connected_ROE
from Deploy import UniformDeploy, DeltaDeploy
from Brigade import Regiment
from Battlefield import Battlefield
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

attack1 = 1
attack_ratio = 2
attack2 = attack1*attack_ratio
Nteam1 = 100
n_ratio = 0.5
Nteam2 = int(Nteam1*n_ratio)
attack_spread = 1
health_spread = 1

Nbattle = 500

def battle_commence(health1_):
    health2 = health1_ * Nteam1 / Nteam2  # make sure total health of both team are the same
    favor = 0
    for _ in range(Nbattle):
        Team1 = Regiment()
        Team2 = Regiment()

        Team1_ROE = fully_connected_ROE()
        Team2_ROE = fully_connected_ROE()
        Team1_Deploy = UniformDeploy(Team1, Nteam1)
        Team2_Deploy = UniformDeploy(Team2, Nteam2)
        Team1_Deploy.deploy(attack1, health1_, attack_spread, health_spread)
        Team2_Deploy.deploy(attack2, health2, attack_spread, health_spread)

        Battle = Battlefield(Team1, Team2)
        Battle.commence(Team1_ROE, Team2_ROE)

        favor += Battle.get_reward()[0]

    return favor/Nbattle

if __name__ == '__main__':
    Ncpu = mp.cpu_count()
    with mp.Pool(processes = Ncpu) as p:
        results = p.map(battle_commence, [health1_ for health1_ in range(4,30)])

    #results = []
    #for health1_ in range(4,20):
    #    result = battle_commence(health1_)
    #    results.append(result)

    plt.plot(list(range(4,30)), results)
    plt.show()


#FILE = open('./data/attack_ratio_%.3f_n_ratio_%.3f.txt'%(attack_ratio, n_ratio),'w')
#FILE.write(str(who_win) + '\n')
#FILE.close()




