import sys
sys.path.append('../')
from Battlefield import Battlefield
from Brigade import Regiment
from Deploy import UniformIntDeploy
from Commander import RandomCommander

import pytest

nbattalion1 = 10
nbattalion2 = 5
attack1, attackspread1, health1, healthspread1 = 5, 1, 10, 2
attack2, attackspread2, health2, healthspread2 = 5, 1, 10, 2

maxbattalion1 = 2
maxbattalion2 = 2

@pytest.fixture
def initialize_regiments():
    regiment1 = Regiment()
    regiment2 = Regiment()
    deploy1 = UniformIntDeploy(regiment1, nbattalion1)
    deploy2 = UniformIntDeploy(regiment2, nbattalion2)
    deploy1.deploy(attack1, health1, attackspread1, healthspread1)
    deploy2.deploy(attack2, health2, attackspread2, healthspread2)
    return regiment1, regiment2

@pytest.fixture
def initalize_randomcommanders():
    commander1 = RandomCommander(maxbattalion1)
    commander2 = RandomCommander(maxbattalion2)
    commander1.set_order_action_map(nbattalion1)
    commander2.set_order_action_map(nbattalion2)
    return commander1, commander2


def test_initialization(initialize_regiments):
    regiment1, regiment2 = initialize_regiments
    assert not regiment1.offense_set
    assert regiment1.battalion_set == set(list(range(nbattalion1)))
    assert not regiment2.offense_set
    assert regiment2.battalion_set == set(list(range(nbattalion2)))

def test_commander(initialize_randomcommanders, initialize_regiments):
    commander1, commander2 = initialize_randomcommanders
    regiment1, regiment2 = initialize_regiments
    battle = Battlefield(regiment1, regiment2)

    order1, ordercode1 = commander1(regiment1, regiment2)
    order2, ordercode2 = commander1(regiment2, regiment1)
    commander1.deliver_order(order1, regiment1)
    commander2.deliver_order(order2, regiment2)

    assert len(regiment1.offense_set) == maxbattalion1*2
    assert len(regiment2.offense_set) == maxbattalion2*2




