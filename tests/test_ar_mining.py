# -*- coding: utf-8 -*-

import numpy as np
import pytest

from src.env.ar_mining import AssociationRuleMining
from src.env.db_loader import load_db
from src.env.ar_mining import Rule


@pytest.fixture
def seed():
    np.random.seed(0)


@pytest.fixture
def env(seed):
    env = AssociationRuleMining(
        sup_delta=0.5,
        conf_delta=0.6,
        data_path="./data/contextIGB.txt",
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
        "./data/contextIGB.txt", 1.0, 0.0, False,
    )
    t = np.array(
        [
            [1, 2, 4, 5],
            [2, 3, 5],
            [1, 2, 4, 5],
            [1, 2, 3, 5],
            [1, 2, 3, 4, 5],
            [2, 3, 4],
        ]
    )
    bmap = np.array(
        [
            [1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
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
        "./data/contextIGB.txt", 0.6, 0.0, False,
    )
    t = np.array([[1, 2, 4, 5], [2, 3, 5], [1, 2, 4, 5]])
    bmap = np.array([[1, 1, 0, 1, 1], [0, 1, 1, 0, 1], [1, 1, 0, 1, 1]])
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
        "./data/contextIGB.txt", 0.4, 0.6, False,
    )
    t = np.array([[1, 2, 3, 5], [1, 2, 3, 4, 5], [2, 3, 4]])
    bmap = np.array([[1, 1, 1, 0, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]])
    b2i = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    i2b = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    assert (
        (transaction == t).all()
        and (bit_map == bmap).all()
        and (b2i_dict == b2i)
        and (i2b_dict == i2b)
    )


def test_env_arg(env):
    assert (
        env.sup_delta,
        env.conf_delta,
        env.plus_reward,
        env.minus_reward,
        env.max_steps,
    ) == (0.5, 0.6, 200, -1, 100,)


def test_env_init(env):
    db_total_support = 6
    min_sup = 3

    assert (env.db_total_support == db_total_support) and (env.min_sup == min_sup)


def test_setup_db(env):
    htwar_1 = set([1, 2, 3, 4, 5])
    b2i_dict = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    i2b_dict = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    bit_map = np.array(
        [
            [1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ]
    )
    prob_bit_stand = np.array([4 / 6, 1.0, 4 / 6, 4 / 6, 5 / 6])

    assert (
        (env.htwar_1 == htwar_1)
        and (env.b2i_dict == b2i_dict)
        and (env.i2b_dict == i2b_dict)
        and np.allclose(env.bit_map, bit_map)
        and np.allclose(env.prob_bit_stand, prob_bit_stand)
    )


def test_step_1(env):
    # empty itemsets = {}
    bv = np.array([0, 0, 0, 0, 0])
    env.bit_vector = bv
    env.step(0)
    assert np.allclose(env.bit_vector, np.array([1, 0, 0, 0, 0]))


def test_step_2(env):
    # empty itemsets = {1}
    bv = np.array([1, 0, 0, 0, 0])
    action = 0
    env.bit_vector = bv
    env.step(action)
    assert np.allclose(env.bit_vector, np.array([0, 0, 0, 0, 0]))


def test_step_3(env):
    # empty itemsets = {}
    bv = np.array([0, 0, 0, 0, 0])
    action = 5
    env.bit_vector = bv
    env.step(action)
    assert not (
        np.allclose(env.bit_vector, np.array([1, 0, 0, 0, 0]))
        or np.allclose(env.bit_vector, np.array([0, 1, 0, 0, 0]))
        or np.allclose(env.bit_vector, np.array([0, 0, 1, 0, 0]))
        or np.allclose(env.bit_vector, np.array([0, 0, 0, 1, 0]))
        or np.allclose(env.bit_vector, np.array([0, 0, 0, 0, 1]))
    )


def test_get_next_sup_conf_1(env):
    # AR itemsets = {2, 3} = sup(4)
    bv = np.array([0, 1, 1, 0, 0])
    log_db = np.log1p(6)
    next_state_support = np.array(
        [
            np.log1p(2) / log_db,
            np.log1p(4) / log_db,
            np.log1p(6) / log_db,
            np.log1p(2) / log_db,
            np.log1p(3) / log_db,
        ]
    )
    next_state_confidence = np.array([2 / 4, 4 / 4, 4 / 6, 2 / 4, 3 / 4])
    sup_conf = np.concatenate([next_state_support, next_state_confidence])
    assert np.isclose(env.get_next_sup_conf(bv), sup_conf).all()


def test_get_next_sup_conf_2(env):
    # empty itemsets = {} = sup(0)
    bv = np.array([0, 0, 0, 0, 0])
    log_db = np.log1p(6)
    next_state_support = np.array(
        [
            np.log1p(4) / log_db,
            np.log1p(6) / log_db,
            np.log1p(4) / log_db,
            np.log1p(4) / log_db,
            np.log1p(5) / log_db,
        ]
    )
    next_state_confidence = np.array([0, 0, 0, 0, 0])
    sup_conf = np.concatenate([next_state_support, next_state_confidence])
    assert np.isclose(env.get_next_sup_conf(bv), sup_conf).all()


def test_get_next_sup_conf_3(env):
    # full itemsets = {1, 2, 3, 4, 5} = sup(1)
    bv = np.array([1, 1, 1, 1, 1])
    log_db = np.log1p(6)
    next_state_support = np.array(
        [
            np.log1p(1) / log_db,
            np.log1p(1) / log_db,
            np.log1p(3) / log_db,
            np.log1p(2) / log_db,
            np.log1p(1) / log_db,
        ]
    )
    next_state_confidence = np.array([1.0, 1.0, 1 / 3, 1 / 2, 1])
    sup_conf = np.concatenate([next_state_support, next_state_confidence])
    assert np.isclose(env.get_next_sup_conf(bv), sup_conf).all()


def test_get_reward_1(env):
    # reward=200 + 4
    # AR itemsets = {2, 4} = sup(4) => {1, 2, 4}
    bv = tuple(np.array([0, 1, 0, 1, 0]))
    bit_action = 0
    assert env._get_reward(bv, bit_action) == 204


def test_get_reward_2(env):
    # reward=200 + 4
    # AR itemsets = {2, 4} = sup(4) => {2, 3, 4}
    bv = tuple(np.array([0, 1, 0, 1, 0]))
    bit_action = 2
    assert env._get_reward(bv, bit_action) == 2


def test_check_pbv_1(env):
    # AR itemsets = {2, 4} = sup(4)
    bv = tuple(np.array([0, 1, 0, 1, 0]))
    assert env._check_pbv(bv)


def test_check_pbv_2(env):
    # UPBV itemsets = {} = sup(0)
    bv = tuple(np.array([0, 0, 0, 0, 0]))
    assert not env._check_pbv(bv)


def test_calc_confidence_1(env):
    # AR itemsets = {2, 4} = sup(4) => {1, 2, 4}
    bv = tuple(np.array([0, 1, 0, 1, 0]))
    bit_action = 0
    assert env._calc_confidence(bv, bit_action) == Rule(
        tuple([2, 4]), tuple([1]), 3, 3 / 4
    )


def test_calc_confidence_2(env):
    # UPBV itemsets = {2, 3, 4, 5} = sup(1) => {2, 4, 5}
    bv = tuple(np.array([0, 1, 1, 1, 1]))
    bit_action = 2
    assert env._calc_confidence(bv, bit_action) == Rule(
        tuple([2, 4, 5]), tuple([3]), 1, 1 / 3
    )


def test_calc_confidence_3(env):
    # empty itemsets = {} = sup(0)
    bv = tuple(np.array([1, 0, 0, 0, 0]))
    bit_action = 0
    assert env._calc_confidence(bv, bit_action) == Rule(tuple([]), tuple([1]), 4, 0)


def test_calc_confidence_4(env):
    # empty itemsets = {} = sup(0)
    bv = tuple(np.array([0, 0, 0, 0, 0]))
    bit_action = 0
    assert env._calc_confidence(bv, bit_action) == Rule(tuple([]), tuple([1]), 4, 0)
