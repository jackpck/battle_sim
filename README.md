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

## Simulation result
Interestingly, quality does not always trump quantity (and vice versa). When the defense of
each soldier is small, the smaller team (more attack and defense per capita) is more likely to win while
when the defense of each soldier is large, the larger team (less attack and defense per capita) is more
likely to win.

