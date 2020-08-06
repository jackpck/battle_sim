from ROE import fully_connected_ROE
from Regiment import Regiment
from Battlefield import Battlefield
import matplotlib.pyplot as plt

attack1, health1 = 2, 10
attack2, health2 = 1, 5
Nteam1 = 10
Nteam2 = 20

Nbattle = 500
bins = 10
trend_list = []
for _ in range(Nbattle):
    trend = 0
    for _ in range(bins):
        Team1 = Regiment(Nteam1)
        Team2 = Regiment(Nteam2)
        Team1.deploy(attack1, health1)
        Team2.deploy(attack2, health2)

        Team1_ROE = fully_connected_ROE(Team1)
        Team2_ROE = fully_connected_ROE(Team2)

        Battle = Battlefield(Team1, Team2, Team1_ROE, Team2_ROE)
        Battle.commence()

        trend += Battle.who_win()
    trend /= bins
    trend_list.append(trend)

print('***** DEBRIEF *****')
print('team1 total attack: {}, team1 total health: {}'.format(Nteam1*attack1,Nteam1*health1))
print('team2 total attack: {}, team2 total health: {}'.format(Nteam2*attack2,Nteam2*health2))
plt.hist(trend_list,bins=20)
plt.show()
