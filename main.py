from ROE import fully_connected_ROE
from Regiment import Regiment
from Battlefield import Battlefield
import numpy as np
import matplotlib.pyplot as plt

attack1 = 1
attack_ratio = 2
attack2 = attack1*attack_ratio
Nteam1 = 20
n_ratio = 0.5
Nteam2 = int(Nteam1*n_ratio)

Nbattle = 500
favors = np.zeros(16)
for i, health1 in enumerate(range(4,20)):
    health2 = health1*Nteam1/Nteam2 # make sure total health of both team are the same

    print('*'*30)
    print('Nteam1: {}, attack1: {}, health1: {}'.format(Nteam1, attack1, health1))
    print('Nteam2: {}, attack2: {}, health2: {}'.format(Nteam2, attack2, health2))
    print('team1 total attack: {}, team1 total health: {}'.format(Nteam1*attack1, Nteam1*health1))
    print('team2 total attack: {}, team2 total health: {}'.format(Nteam2*attack2, Nteam2*health2))

    #FILE = open('./data/attack_ratio_%.3f_n_ratio_%.3f.txt'%(attack_ratio, n_ratio),'w')
    for _ in range(Nbattle):
        Team1 = Regiment(Nteam1)
        Team2 = Regiment(Nteam2)
        Team1.deploy(attack1, health1)
        Team2.deploy(attack2, health2)

        Team1_ROE = fully_connected_ROE(Team1)
        Team2_ROE = fully_connected_ROE(Team2)

        Battle = Battlefield(Team1, Team2, Team1_ROE, Team2_ROE)
        Battle.commence()

        who_win = Battle.who_win()
        favors[i] += who_win
        #FILE.write(str(who_win) + '\n')

    favors[i] = favors[i]/Nbattle

plt.plot(favors)
plt.show()
    #FILE.close()




