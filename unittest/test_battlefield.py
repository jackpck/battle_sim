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
    battle = Battlefield(regiment1, regiment2)
    return regiment1, regiment2, battle

@pytest.fixture
def initialize_randomcommanders():
    regiment1 = Regiment()
    regiment2 = Regiment()
    deploy1 = UniformIntDeploy(regiment1, nbattalion1)
    deploy2 = UniformIntDeploy(regiment2, nbattalion2)
    deploy1.deploy(attack1, health1, attackspread1, healthspread1)
    deploy2.deploy(attack2, health2, attackspread2, healthspread2)
    battle = Battlefield(regiment1, regiment2)

    commander1 = RandomCommander(regiment1, regiment2, maxbattalion1)
    commander2 = RandomCommander(regiment2, regiment1, maxbattalion2)
    commander1.set_order_action_map()
    commander2.set_order_action_map()
    return commander1, commander2, battle


def test_initialization(initialize_regiments):
    regiment1, regiment2, battle = initialize_regiments

    assert not regiment1.offense_set
    assert regiment1.battalion_set == set(list(range(nbattalion1)))
    assert not regiment2.offense_set
    assert regiment2.battalion_set == set(list(range(nbattalion2)))
    assert len(battle.state) == regiment1.get_full_size() + regiment2.get_full_size()

    for i in range(nbattalion1):
        assert regiment1.battalions[i].get_target() is None


def test_commander(initialize_randomcommanders):
    commander1, commander2, battle = initialize_randomcommanders
    order1, ordercode1 = commander1.order(battle.state)
    order2, ordercode2 = commander2.order(battle.state)
    commander1.deliver_order(order1)
    commander2.deliver_order(order2)

    assert len(battle.get_regiment1().offense_set) == commander1.nbatcommand
    assert len(battle.get_regiment2().offense_set) == commander2.nbatcommand
    assert battle.get_regiment1().battalions[order1[0]].get_target() == order1[commander1.nbatcommand]
    assert battle.get_regiment1().battalions[order1[1]].get_target() == order1[commander1.nbatcommand + 1]
    assert battle.get_regiment2().battalions[order2[0]].get_target() == order2[commander2.nbatcommand]
    assert battle.get_regiment2().battalions[order2[1]].get_target() == order2[commander2.nbatcommand + 1]

def test_commander_none(initialize_randomcommanders):
    '''
    Test initial targets are set to None
    Test if order is set to None, target will be None.
    '''
    commander1, commander2, battle = initialize_randomcommanders

    for i in range(nbattalion1):
        assert battle.get_regiment1().battalions[i].get_target() is None

    order1, ordercode1 = commander1.order(battle.state)
    order2, ordercode2 = commander2.order(battle.state)
    order1[0] = None
    order1[1] = None
    commander1.deliver_order(order1)
    commander2.deliver_order(order2)

    for i in range(nbattalion1):
        assert battle.get_regiment1().battalions[i].get_target() is None

def test_two_attacks_on_same_defender(initialize_randomcommanders):
    '''
    Test dead battalion cannot attack if chosen
    '''
    commander1, commander2, battle = initialize_randomcommanders

    order1, ordercode1 = commander1.order(battle.state)
    order2, ordercode2 = commander2.order(battle.state)
    order1[0] = 2
    order1[1] = 4
    order1[2] = 3
    order1[3] = 3
    commander1.deliver_order(order1)
    commander2.deliver_order(order2)

    battalion1_attack2 = battle.get_regiment1().battalions[2].get_attack()
    battalion1_attack4 = battle.get_regiment1().battalions[4].get_attack()
    battalion2_health3 = battle.get_regiment2().battalions[3].get_health()
    battle.get_regiment1().fire(battle.get_regiment2())
    battle.get_regiment2().fire(battle.get_regiment1())

    assert battalion2_health3 - battalion1_attack2 - battalion1_attack4 == \
           battle.get_regiment2().battalions[3].get_health()


    commander1, commander2, battle = initialize_randomcommanders
    battle.get_regiment1().battalions[2].set_health(-1)
    battle.get_regiment1().count_KIA()
    battle.update_state()

    order1, ordercode1 = commander1.order(battle.state)
    order2, ordercode2 = commander2.order(battle.state)
    order1[0] = 2
    order1[1] = 4
    order1[2] = 3
    order1[3] = 3
    commander1.deliver_order(order1)
    commander2.deliver_order(order2)

    battalion1_attack4 = battle.get_regiment1().battalions[4].get_attack()
    battalion2_health3 = battle.get_regiment2().battalions[3].get_health()
    battle.get_regiment1().fire(battle.get_regiment2())
    battle.get_regiment2().fire(battle.get_regiment1())

    assert battalion2_health3 - battalion1_attack4 == \
           battle.get_regiment2().battalions[3].get_health()


