#WARGAME
## Does quality trump quantity?
Is it better to bring more less-equiped men or less well-equiped men to a battle? In this 
simulation, I will look at the result of two teams battling out: one team with twice as 
many men as the other team but less attack then the other team. In order to hold everything
else constant, the total defense and the total attack of both team will be the same.

## Rules of Engagement
### Vanilla
A surviving soldier from each team is paired up randomly. Each will delivery an attack to
his enemy. The enemy, upon receiving the damage, will have his defense value reduced by
the attacker's attack value. If the defense of a soldier becomes equal to or less than zero,
that soldier is removed from the battlefield and will no longer participate in future rounds.
A team is declared winner when there is at least one surviving soldier while the other team
has none.

### With reservists
Instead of having all available soldiers attack, save half of the soldiers behind the 
frontline as reservists. A reservist cannot attack unless the a frontline soldier is KIA.

## Simulation result
Interestingly, quality does not always trump quantity (and vice versa). When the team size
is small, the larger team (less attack and defense per capita) is more likely to win while
when the team size is large, the smaller team (more attack and defense per capita) is more
likely to win.

The implication can be very interesting. Say both teams are large. If they battle out, we know
the smaller-but-better team will win. However, if each team is to be divided into subgroups
e.g. one army into ten regiments, and each subgroup can only engage with a fixed 
subgroup of the other team. Then as long as the size of the subgroup is small enough, the 
larger team will win!
