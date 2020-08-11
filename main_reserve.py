from ROE import fully_connected_ROE
from Deploy import UniformDeploy, DeltaDeploy
from Regiment import Regiment
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
noise = 1
Nreserve = 20

Nbattle = 500
Ncpu = mp.cpu_count()

def battle_commence(health1_):
    health2 = health1_ * Nteam1 / Nteam2  # make sure total health of both team are the same
    N = Nteam2 # size of frontline and reserve
    favor = 0
    for _ in range(Nbattle):
        Team1 = Regiment(2*N-Nreserve)
        Team1_Reserve = Regiment(Nreserve)
        Team2 = Regiment(N)

        Team1_ROE = fully_connected_ROE(Team1)
        Team2_ROE = fully_connected_ROE(Team2)
        Team1_Deploy = UniformDeploy(Team1)
        Team1_Reserve_Deploy = UniformDeploy(Team1_Reserve)
        Team2_Deploy = UniformDeploy(Team2)
        Team1_Deploy.deploy(attack1, health1_, noise)
        Team1_Reserve_Deploy.deploy(attack1, health1_, noise)
        Team2_Deploy.deploy(attack2, health2, noise)

        Battle = Battlefield(Team1, Team2, Team1_ROE, Team2_ROE)
        Battle.commence_with_reinforcement(Team1_Reserve)

        favor += Battle.who_win()

    return favor/Nbattle

if __name__ == '__main__':
    import sys
    import pandas as pd

    health_values = list(range(4,30))
    with mp.Pool(processes = Ncpu) as p:
        results = p.map(battle_commence, [health1_ for health1_ in health_values])

    #results = []
    #for health1_ in health_values:
    #    result = battle_commence(health1_,Nreserve)
    #    results.append(result)

    df = pd.DataFrame({'health1':health_values, 'favor':results})
    df.to_csv('./data/reserve_%i.csv'%Nreserve)
    plt.plot(health_values, results)
    plt.show()




#FILE = open('./data/attack_ratio_%.3f_n_ratio_%.3f.txt'%(attack_ratio, n_ratio),'w')
#FILE.write(str(who_win) + '\n')
#FILE.close()




