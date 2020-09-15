# WARGAME (Reinforcement learning)
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
has none. The rule of engagement is defined in `ROE.py`.

## Simulation result
Interestingly, quality does not always trump quantity (and vice versa). When the defense of
each soldier is small, the smaller team (more attack and defense per capita) is more likely to win while
when the defense of each soldier is large, the larger team (less attack and defense per capita) is more
likely to win. The simulation is run in `main_simulation.py`.

## Reinforcement learning
Can we teach a machine to micromanage the units and do better than just randomly choosing enemy to attack?
Next I will train a deep Q-learning model to play the wargame. One of the reasons for using a deep Q-learning
instead of a (tabulated) Q-learning model is that the state space (health of the soldiers ) is continuous here.
A neural network is used to learn the Q-value of each actions given the state of the battle. 

The number of possible actions is huge if all soldier are free to choose their targets. Therefore I will first
limit the deep Q neural network (DQN) to learn the strategy of a battle that involves only 5 soldiers on each side
and only one soldier per round is chosen to attack its target. Again, whichever side first loses all its soldiers 
loses the battle. The DQN is trained in `main_train_dqn.py`.
