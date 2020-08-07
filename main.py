from ROE import fully_connected_ROE
from Regiment import Regiment
from Battlefield import Battlefield
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

attack1 = 1
attack_ratio = 1.9
attack2 = attack1*attack_ratio
Nteam1 = 60
n_ratio = 0.5
Nteam2 = int(Nteam1*n_ratio)

Nbattle = 500
Ncpu = mp.cpu_count()

def battle_commence(health1_):
    health2 = health1_ * Nteam1 / Nteam2  # make sure total health of both team are the same
    favor = 0
    for _ in range(Nbattle):
        Team1 = Regiment(Nteam1)
        Team2 = Regiment(Nteam2)
        Team1.deploy(attack1, health1_)
        Team2.deploy(attack2, health2)

        Team1_ROE = fully_connected_ROE(Team1)
        Team2_ROE = fully_connected_ROE(Team2)

        Battle = Battlefield(Team1, Team2, Team1_ROE, Team2_ROE)
        Battle.commence()

        favor += Battle.who_win()

    return favor/Nbattle

with mp.Pool(processes = Ncpu) as p:
    results = p.map(battle_commence, [health1_ for health1_ in range(4,20)])


plt.plot(list(range(4,20)), results)
plt.show()




#FILE = open('./data/attack_ratio_%.3f_n_ratio_%.3f.txt'%(attack_ratio, n_ratio),'w')
#FILE.write(str(who_win) + '\n')
#FILE.close()




