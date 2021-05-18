# -*- coding: utf-8 -*-

import numpy as np
import pytest

from src.env.fp_mining import FrequentPatternMining
from src.env.db_loader import load_db
from src.env.fp_mining import Pattern


@pytest.fixture
def seed():
    np.random.seed(0)


@pytest.fixture
def env(seed):
    env = FrequentPatternMining(
        delta=0.4,
        data_path="./data/contextPasquier99.txt",
        used=1.0,
        head=0.0,
        shuffle_db=False,
        plus_reward=200,
        minus_reward=-1,
        max_steps=100,
        cache_limit=10000,
    )
    return env


def test_load_db_1():
    """
    Properly load the dataset
    """
    transaction, bit_map, b2i_dict, i2b_dict = load_db(
        "./data/contextPasquier99.txt", 1.0, 0.0, False,
    )
    t = np.array([[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5], [1, 2, 3, 5]])
    bmap = np.array(
        [
            [1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 1, 1, 0, 1],
            [0, 1, 0, 0, 1],
            [1, 1, 1, 0, 1],
        ]
    )
    b2i = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    i2b = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    assert (
        (transaction == t).all()
        and (bit_map == bmap).all()
        and (b2i_dict == b2i)
        and (i2b_dict == i2b)
    )


def test_load_db_2():
    """
    Properly load the dataset(60%)
    """
    transaction, bit_map, b2i_dict, i2b_dict = load_db(
        "./data/contextPasquier99.txt", 0.6, 0.0, False,
    )
    t = np.array([[1, 3, 4], [2, 3, 5], [1, 2, 3, 5]])
    bmap = np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1], [1, 1, 1, 0, 1]])
    b2i = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    i2b = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    assert (
        (transaction == t).all()
        and (bit_map == bmap).all()
        and (b2i_dict == b2i)
        and (i2b_dict == i2b)
    )


def test_load_db_3():
    """
    Properly load the dataset(the other 40%)
    """
    transaction, bit_map, b2i_dict, i2b_dict = load_db(
        "./data/contextPasquier99.txt", 0.4, 0.6, False,
    )
    t = np.array([[2, 5], [1, 2, 3, 5]])
    bmap = np.array([[0, 1, 0, 1], [1, 1, 1, 1]])
    b2i = {0: 1, 1: 2, 2: 3, 3: 5}
    i2b = {1: 0, 2: 1, 3: 2, 5: 3}
    assert (
        (transaction == t).all()
        and (bit_map == bmap).all()
        and (b2i_dict == b2i)
        and (i2b_dict == i2b)
    )


def test_env_arg(env):
    assert (env.delta, env.plus_reward, env.minus_reward, env.max_steps) == (
        0.4,
        200,
        -1,
        100,
    )


def test_env_init(env):
    db_total_support = 5
    min_sup = 2
    each_htwfp_support = np.array([3, 4, 4, 4])

    assert (
        (env.db_total_support == db_total_support)
        and (env.min_sup == min_sup)
        and np.allclose(env.each_htwfp_support, each_htwfp_support)
    )


def test_setup_db(env):
    htwsp_1 = set([1, 2, 3, 5])
    b2i_dict = {0: 1, 1: 2, 2: 3, 3: 5}
    i2b_dict = {1: 0, 2: 1, 3: 2, 5: 3}
    bit_map = np.array(
        [[1, 0, 1, 0], [0, 1, 1, 1], [1, 1, 1, 1], [0, 1, 0, 1], [1, 1, 1, 1]]
    )
    prob_bit_stand = np.array([0.6, 0.8, 0.8, 0.8])

    assert (
        (env.htwsp_1 == htwsp_1)
        and (env.b2i_dict == b2i_dict)
        and (env.i2b_dict == i2b_dict)
        and np.allclose(env.bit_map, bit_map)
        and np.allclose(env.prob_bit_stand, prob_bit_stand)
    )


def test_step_1(env):
    # empty itemsets = {}
    bv = np.array([0, 0, 0, 0])
    env.bit_vector = bv
    env.step(0)
    assert np.allclose(env.bit_vector, np.array([1, 0, 0, 0]))


def test_step_2(env):
    # empty itemsets = {1}
    bv = np.array([1, 0, 0, 0])
    action = 0
    env.bit_vector = bv
    env.step(action)
    assert np.allclose(env.bit_vector, np.array([0, 0, 0, 0]))


def test_step_3(env):
    # empty itemsets = {}
    bv = np.array([0, 0, 0, 0])
    action = 4
    env.bit_vector = bv
    env.step(action)
    assert not (
        np.allclose(env.bit_vector, np.array([1, 0, 0, 0]))
        or np.allclose(env.bit_vector, np.array([0, 1, 0, 0]))
        or np.allclose(env.bit_vector, np.array([0, 0, 1, 0]))
        or np.allclose(env.bit_vector, np.array([0, 0, 0, 1]))
    )


def test_get_next_state_support_1(env):
    # FP itemsets = {1, 3} = sup(3)
    bv = np.array([1, 0, 1, 0])
    log_db = np.log1p(5)
    next_state_support = np.array(
        [
            np.log1p(4) / log_db,
            np.log1p(2) / log_db,
            np.log1p(3) / log_db,
            np.log1p(2) / log_db,
        ]
    )
    assert np.isclose(env.get_next_state_support(bv), next_state_support).all()


def test_get_next_state_support_2(env):
    # empty itemsets = {} = sup(0)
    bv = np.array([0, 0, 0, 0])
    log_db = np.log1p(5)
    next_state_support = np.array(
        [
            np.log1p(3) / log_db,
            np.log1p(4) / log_db,
            np.log1p(4) / log_db,
            np.log1p(4) / log_db,
        ]
    )
    assert np.isclose(env.get_next_state_support(bv), next_state_support).all()


def test_get_next_state_support_3(env):
    # full itemsets = {1, 2, 3, 5} = sup(0)
    bv = np.array([1, 1, 1, 1])
    log_db = np.log1p(5)
    next_state_support = np.array(
        [
            np.log1p(3) / log_db,
            np.log1p(2) / log_db,
            np.log1p(2) / log_db,
            np.log1p(2) / log_db,
        ]
    )
    assert np.isclose(env.get_next_state_support(bv), next_state_support).all()


def test_get_reward_1(env):
    # reward=200 + 4
    # FP itemsets = {2, 5} = sup(4)
    bv = tuple(np.array([0, 1, 0, 1]))
    assert env._get_reward(bv) == 204


def test_get_reward_2(env):
    # reward=-1
    # UPBV itemsets = {} = sup(0)
    bv = tuple(np.array([0, 0, 0, 0]))
    assert env._get_reward(bv) == -1


def test_get_support_reward_1(env):
    # FP itemsets = {2, 5} = sup(4)
    bv = tuple(np.array([0, 1, 0, 1]))
    assert env._get_support_reward(bv) == 4


def test_get_support_reward_2(env):
    # UPBV itemsets = {} = sup(0)
    bv = tuple(np.array([0, 0, 0, 0]))
    assert env._get_support_reward(bv) == 0


def test_check_pbv_1(env):
    # FP itemsets = {2, 5} = sup(4)
    bv = tuple(np.array([0, 1, 0, 1]))
    assert env._check_pbv(bv)


def test_check_pbv_2(env):
    # UPBV itemsets = {} = sup(0)
    bv = tuple(np.array([0, 0, 0, 0]))
    assert not env._check_pbv(bv)


def test_calc_support_1(env):
    # FP itemsets = {2, 5} = sup(4)
    bv = tuple(np.array([0, 1, 0, 1]))
    assert env._calc_support(bv) == Pattern(tuple([2, 5]), 4)


def test_calc_support_2(env):
    # UPBV itemsets = {1} = sup(3)
    bv = tuple(np.array([1, 0, 0, 0]))
    assert env._calc_support(bv) == Pattern(tuple([1]), 3)


def test_calc_support_3(env):
    # empty itemsets = {} = sup(0)
    bv = tuple(np.array([0, 0, 0, 0]))
    assert env._calc_support(bv) == Pattern(tuple([]), 0)


def test_calc_support_4(env):
    # full itemsets = {1, 2, 3, 5} = sup(0)
    bv = tuple(np.array([1, 1, 1, 1]))
    assert env._calc_support(bv) == Pattern(tuple([1, 2, 3, 5]), 2)
