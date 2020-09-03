from ROE import fully_connected_ROE
from Deploy import UniformDeploy, DeltaDeploy
from Brigade import Brigade
from Battlefield import Battlefield
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

attack1 = 1
attack_ratio = 2
attack2 = attack1*attack_ratio
n_ratio = 0.5
noise = 1
reserve_fraction = 0.0
health1 = 7
Nbattle = 500
Ncpu = mp.cpu_count()

def battle_commence(Nteam1):
    Nteam2 = int(Nteam1 * n_ratio)
    health2 = health1 * Nteam1 / Nteam2  # make sure total health of both team are the same
    favor = 0
    for _ in range(Nbattle):
        Team1 = Brigade(Nteam1)
        Team2 = Brigade(Nteam2)

        Team1_ROE = fully_connected_ROE(Team1)
        Team2_ROE = fully_connected_ROE(Team2)
        Team1_Deploy = UniformDeploy(Team1)
        Team2_Deploy = UniformDeploy(Team2)
        Team1_Deploy.deploy(attack1, health1, noise, reserve=reserve_fraction)
        Team2_Deploy.deploy(attack2, health2, noise)

        Battle = Battlefield(Team1, Team2, Team1_ROE, Team2_ROE)
        Battle.commence(use_res1=False, use_res2=False)

        favor += Battle.who_win()

    return favor/Nbattle

if __name__ == '__main__':
    import sys
    import pandas as pd

    Nteam1_value = list(range(10,210,10))
    with mp.Pool(processes = Ncpu) as p:
        results = p.map(battle_commence, [Nteam1_ for Nteam1_ in Nteam1_value])

    #results = []
    #for health1_ in health_values:
    #    result = battle_commence(health1_)
    #    results.append(result)

    df = pd.DataFrame({'Nteam1':Nteam1_value, 'favor':results})
    df.to_csv('./data/health1_%.1f_reserve_fraction_%.3f.csv'%(health1,reserve_fraction))
    plt.plot(Nteam1_value, results)
    plt.show()

