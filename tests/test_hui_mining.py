# -*- coding: utf-8 -*-

import numpy as np
import pytest

from src.env.hui_mining import HighUtilityItemsetsMining
from src.env.db_loader import load_hui_db
from src.env.hui_mining import Rule


@pytest.fixture
def seed():
    np.random.seed(0)


@pytest.fixture
def env(seed):
    env = HighUtilityItemsetsMining(
        delta=0.3125,
        data_path="./data/DB_Utility.txt",
        used=1.0,
        head=0.0,
        shuffle_db=False,
        plus_reward=200,
        minus_reward=-1,
        max_steps=100,
        cache_limit=10000,
    )
    return env


def test_load_hui_db_1():
    """
    Properly load the dataset
    """
    database, items, utils = load_hui_db("./data/DB_Utility.txt", 1.0, 0.0, False,)
    db = np.array(
        [
            {3: 1, 5: 3, 1: 5, 2: 10, 4: 6, 6: 5},
            {3: 3, 5: 3, 2: 8, 4: 6},
            {3: 1, 1: 5, 4: 2},
            {3: 6, 5: 6, 1: 10, 7: 5},
            {3: 2, 5: 3, 2: 4, 7: 2},
        ]
    )
    i = set([1, 2, 3, 4, 5, 6, 7])
    u = np.array([30, 20, 8, 27, 11])
    assert (database == db).all() and items == i and (utils == u).all()


def test_load_hui_db_2():
    """
    Properly load the dataset(60%)
    """
    database, items, utils = load_hui_db("./data/DB_Utility.txt", 0.6, 0.0, False,)
    db = np.array(
        [
            {3: 1, 5: 3, 1: 5, 2: 10, 4: 6, 6: 5},
            {3: 3, 5: 3, 2: 8, 4: 6},
            {3: 1, 1: 5, 4: 2},
        ]
    )
    i = set([1, 2, 3, 4, 5, 6])
    u = np.array([30, 20, 8])
    assert (database == db).all() and items == i and (utils == u).all()


def test_load_hui_db_3():
    """
    Properly load the dataset(the other 40%)
    """
    database, items, utils = load_hui_db("./data/DB_Utility.txt", 0.4, 0.6, False,)
    db = np.array([{3: 6, 5: 6, 1: 10, 7: 5}, {3: 2, 5: 3, 2: 4, 7: 2}])
    i = set([1, 2, 3, 5, 7])
    u = np.array([27, 11])
    assert (database == db).all() and items == i and (utils == u).all()


def test_env_arg(env):
    assert (env.delta, env.plus_reward, env.minus_reward, env.max_steps) == (
        0.3125,
        200,
        -1,
        100,
    )


def test_env_init(env):
    db_total_util = 96
    min_util = 30
    each_htwui_utility = np.array([20, 22, 13, 14, 15, 5, 7])

    assert (
        (env.db_total_util == db_total_util)
        and (env.min_util == min_util)
        and (env.each_htwui_utility == each_htwui_utility).all()
    )


def test_setup_db(env):
    htwui_1 = set([1, 2, 3, 4, 5, 6, 7])
    b2i_dict = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}
    i2b_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
    bit_map = np.array(
        [
            [5, 10, 1, 6, 3, 5, 0],
            [0, 8, 3, 6, 3, 0, 0],
            [5, 0, 1, 2, 0, 0, 0],
            [10, 0, 6, 0, 6, 0, 5],
            [0, 4, 2, 0, 3, 0, 2],
        ]
    )
    prob_bit_stand = np.array([0.6, 0.6, 1.0, 0.6, 0.8, 0.2, 0.4])

    assert (
        (env.htwui_1 == htwui_1)
        and (env.b2i_dict == b2i_dict)
        and (env.i2b_dict == i2b_dict)
        and (env.bit_map == bit_map).all()
        and (env.prob_bit_stand == prob_bit_stand).all()
    )


def test_step_1(env):
    # empty itemsets = {}
    bv = np.array([0, 0, 0, 0, 0, 0, 0])
    env.bit_vector = bv
    env.step(0)
    assert (env.bit_vector == np.array([1, 0, 0, 0, 0, 0, 0])).all()


def test_step_2(env):
    # empty itemsets = {1}
    bv = np.array([1, 0, 0, 0, 0, 0, 0])
    action = 0
    env.bit_vector = bv
    env.step(action)
    assert (env.bit_vector == np.array([0, 0, 0, 0, 0, 0, 0])).all()


def test_step_3(env):
    # empty itemsets = {}
    bv = np.array([0, 0, 0, 0, 0, 0, 0])
    action = 7
    env.bit_vector = bv
    env.step(action)
    assert not (
        np.allclose(env.bit_vector, np.array([1, 0, 0, 0, 0, 0, 0]))
        or np.allclose(env.bit_vector, np.array([0, 1, 0, 0, 0, 0, 0]))
        or np.allclose(env.bit_vector, np.array([0, 0, 1, 0, 0, 0, 0]))
        or np.allclose(env.bit_vector, np.array([0, 0, 0, 1, 0, 0, 0]))
        or np.allclose(env.bit_vector, np.array([0, 0, 0, 0, 1, 0, 0]))
        or np.allclose(env.bit_vector, np.array([0, 0, 0, 0, 0, 1, 0]))
        or np.allclose(env.bit_vector, np.array([0, 0, 0, 0, 0, 0, 1]))
    )


def test_get_next_state_utility_1(env):
    # HUI itemsets = {2, 5} = util(31)
    bv = np.array([0, 1, 0, 0, 1, 0, 0])
    log_db = np.log1p(96)
    next_state_utility = np.array(
        [
            np.log1p(18) / log_db,
            np.log1p(15) / log_db,
            np.log1p(37) / log_db,
            np.log1p(36) / log_db,
            np.log1p(22) / log_db,
            np.log1p(18) / log_db,
            np.log1p(9) / log_db,
        ]
    )
    assert np.isclose(env.get_next_state_utility(bv), next_state_utility).all()


def test_get_next_state_utility_2(env):
    # empty itemsets = {} = util(0)
    bv = np.array([0, 0, 0, 0, 0, 0, 0])
    log_db = np.log1p(96)
    next_state_utility = np.array(
        [
            np.log1p(20) / log_db,
            np.log1p(22) / log_db,
            np.log1p(13) / log_db,
            np.log1p(14) / log_db,
            np.log1p(15) / log_db,
            np.log1p(0) / log_db,
            np.log1p(0) / log_db,
        ]
    )
    assert np.isclose(env.get_next_state_utility(bv), next_state_utility).all()


def test_get_next_state_utility_3(env):
    # full itemsets = {1, 2, 3, 4, 5, 6, 7} = util(0)
    bv = np.array([1, 1, 1, 1, 1, 1, 1])
    log_db = np.log1p(96)
    next_state_utility = np.array(
        [
            np.log1p(0) / log_db,
            np.log1p(0) / log_db,
            np.log1p(0) / log_db,
            np.log1p(0) / log_db,
            np.log1p(0) / log_db,
            np.log1p(0) / log_db,
            np.log1p(30) / log_db,
        ]
    )
    assert np.isclose(env.get_next_state_utility(bv), next_state_utility).all()


def test_get_reward_1(env):
    # reward=200 + 4
    # HUI itemsets = {2, 5} = util(31)
    bv = tuple(np.array([0, 1, 0, 0, 1, 0, 0]))
    assert env._get_reward(bv) == 204


def test_get_reward_2(env):
    # reward=-1
    # UPBV itemsets = {4, 6, 7} = util(0)
    bv = tuple(np.array([0, 0, 0, 1, 0, 1, 1]))
    assert env._get_reward(bv) == -1


def test_get_reward_3(env):
    # reward=1
    # 1/4 util itemsets = {3} = util(13)
    bv = tuple(np.array([0, 0, 1, 0, 0, 0, 0]))
    assert env._get_reward(bv) == 1


def test_get_utility_reward_1(env):
    # HUI itemsets = {2, 5} = util(31)
    bv = tuple(np.array([0, 1, 0, 0, 1, 0, 0]))
    assert env._get_utility_reward(bv) == 4


def test_get_utility_reward_2(env):
    # 3/4 util itemsets = {1, 3} = util(28)
    bv = tuple(np.array([1, 0, 1, 0, 0, 0, 0]))
    assert env._get_utility_reward(bv) == 3


def test_get_utility_reward_3(env):
    # 1/2 util itemsets = {1} = util(20)
    bv = tuple(np.array([1, 0, 0, 0, 0, 0, 0]))
    assert env._get_utility_reward(bv) == 2


def test_get_utility_reward_4(env):
    # 1/4 util itemsets = {3} = util(13)
    bv = tuple(np.array([0, 0, 1, 0, 0, 0, 0]))
    assert env._get_utility_reward(bv) == 1


def test_get_utility_reward_5(env):
    # UPBV itemsets = {4, 6, 7} = util(0)
    bv = tuple(np.array([0, 0, 0, 1, 0, 1, 1]))
    assert env._get_utility_reward(bv) == 0


def test_check_pbv_1(env):
    # HUI itemsets = {2, 5} = util(31)
    bv = tuple(np.array([0, 1, 0, 0, 1, 0, 0]))
    assert env._check_pbv(bv)


def test_check_pbv_2(env):
    # UPBV itemsets = {4, 6, 7} = util(0)
    bv = tuple(np.array([0, 0, 0, 1, 0, 1, 1]))
    assert not env._check_pbv(bv)


def test_check_pbv_3(env):
    # UPBV itemsets = {} = util(0)
    bv = tuple(np.array([0, 0, 0, 0, 0, 0, 0]))
    assert not env._check_pbv(bv)


def test_calc_utility_1(env):
    # HUI itemsets = {2, 5} = util(31)
    bv = tuple(np.array([0, 1, 0, 0, 1, 0, 0]))
    assert env._calc_utility(bv) == Rule(tuple([2, 5]), 31)


def test_calc_utility_2(env):
    # UPBV itemsets = {4, 6, 7} = util(0)
    bv = tuple(np.array([0, 0, 0, 1, 0, 1, 1]))
    assert env._calc_utility(bv) == Rule(tuple([4, 6, 7]), 0)


def test_calc_utility_3(env):
    # empty itemsets = {} = util(0)
    bv = tuple(np.array([0, 0, 0, 0, 0, 0, 0]))
    assert env._calc_utility(bv) == Rule(tuple([]), 0)


def test_calc_utility_4(env):
    # full itemsets = {1, 2, 3, 4, 5, 6, 7} = util(0)
    bv = tuple(np.array([1, 1, 1, 1, 1, 1, 1]))
    assert env._calc_utility(bv) == Rule(tuple([1, 2, 3, 4, 5, 6, 7]), 0)


def test_calc_utility_5(env):
    # 1/4 util itemsets = {3} = util(13)
    bv = tuple(np.array([0, 0, 1, 0, 0, 0, 0]))
    assert env._calc_utility(bv) == Rule(tuple([3]), 13)


def test_calc_utility_6(env):
    # 1/2 util itemsets = {1} = util(20)
    bv = tuple(np.array([1, 0, 0, 0, 0, 0, 0]))
    assert env._calc_utility(bv) == Rule(tuple([1]), 20)


def test_calc_utility_7(env):
    # 3/4 util itemsets = {1, 3} = util(28)
    bv = tuple(np.array([1, 0, 1, 0, 0, 0, 0]))
    assert env._calc_utility(bv) == Rule(tuple([1, 3]), 28)


def test_is_utility_gt_quarter_1(env):
    # HUI itemsets = {2, 5} = util(31)
    bv = tuple(np.array([0, 1, 0, 0, 1, 0, 0]))
    assert env._is_utility_gt_quarter(bv)


def test_is_utility_gt_quarter_2(env):
    # util itemsets = {7} = util(7)
    bv = tuple(np.array([0, 0, 0, 0, 0, 0, 1]))
    assert not env._is_utility_gt_quarter(bv)
