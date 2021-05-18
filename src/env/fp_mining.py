# -*- coding: utf-8 -*-

from collections import namedtuple
from functools import lru_cache
import gym
import gym.spaces
from itertools import count
import numpy as np
import sys

from src.env.db_loader import load_db


LIMIT = 100000
Pattern = namedtuple("Pattern", ("itemsets", "support"))


class FrequentPatternMining(gym.Env):
    def __init__(
        self,
        delta=0.3,
        data_path="data/chess.txt",
        used=1.0,
        head=0.0,
        shuffle_db=False,
        plus_reward=1,
        minus_reward=-1,
        max_steps=1000,
        cache_limit=100000,
    ):
        super().__init__()
        self.transaction, self.bit_map, self.b2i_dict, self.i2b_dict = load_db(
            data_path, used, head, shuffle_db
        )
        # Specify the maximum LRU cache as a global variable
        global LIMIT
        LIMIT = cache_limit
        # Setting minimum support threshold
        self.delta = delta
        self.db_total_support = self.bit_map.shape[0]
        self.min_sup = self.delta * self.db_total_support

        self._setup_db()
        self.max_steps = max_steps
        # Action = invert each bit + regenerate bit vector
        self.action_space = gym.spaces.Discrete(len(self.htwsp_1) + 1)
        # Observation = each support
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=((len(self.htwsp_1)),)
        )
        self.plus_reward = plus_reward
        self.minus_reward = minus_reward
        self.reward_range = [float(minus_reward), float(plus_reward)]
        self.done = False
        self.steps = 0
        self.num_random_bv = 0
        self.total_reward = 0
        # Calculate the support of each item
        self.each_htwfp_support = np.sum(self.bit_map, axis=0)
        self.reset()

    def _setup_db(self):
        # high transaction-weighted support pattern
        self.htwsp_1 = set(
            self.b2i_dict[i]
            for i in np.where(np.sum(self.bit_map, axis=0) >= self.min_sup)[0]
        )
        self.b2i_dict = {i: v for i, v in enumerate(self.htwsp_1)}
        self.i2b_dict = {v: i for i, v in enumerate(self.htwsp_1)}
        self.bit_map = np.zeros(
            (self.bit_map.shape[0], len(self.htwsp_1)), dtype=np.int
        )
        self.bit_vector = None  # bv
        for i, t in enumerate(self.transaction):
            for item in t:
                if item in self.htwsp_1:
                    self.bit_map[i][self.i2b_dict[item]] = 1

        # Probability that each item appears on the DB
        self.prob_bit_stand = np.sum(self.bit_map, axis=0) / self.bit_map.shape[0]
        # Set of Frequents Itemset
        self.sfp = set()

    def reset(self):
        if self.bit_vector is None or not self._check_pbv(tuple(self.bit_vector)):
            self.bit_vector = self._create_random_pbv()

        self.done = False
        self.steps = 0
        self.sfp = set()
        self.num_random_bv = 0
        return self._observe()

    def step(self, action):
        # The last index of the action is the regeneration of the bit-vector
        if action == len(self.htwsp_1):
            self.bit_vector = self._create_random_pbv()
        else:
            # Bit inversion
            self.bit_vector[action] = 1 if self.bit_vector[action] == 0 else 0
        self.steps += 1
        observation = self._observe()
        reward = self._get_reward(self.bit_vector)
        self.done = self._is_done()
        return observation, reward, self.done, {}

    def render(self):
        result = []
        for fp in self.sfp:
            itemsets = " ".join(map(lambda x: str(int(x)), fp.itemsets))
            result.append("{} #SUP: {}".format(itemsets, fp.support))
        with open("result.txt", mode="w") as f:
            f.write("\n".join(result))

    def get_next_state_support(self, bv):
        next_state_support = np.zeros(len(self.htwsp_1))
        for i in range(len(self.htwsp_1)):
            # Invert each bit and check if it is a PBV.
            # Reduce the amount of computation by changing the bit-vector directly.
            prev_i = bv[i]
            bv[i] = 1 if bv[i] == 0 else 0
            if not self._check_pbv((tuple(bv))):
                bv[i] = prev_i
                continue
            pattern = self._calc_support((tuple(bv)))
            next_state_support[i] = np.log1p(pattern.support) / np.log1p(
                self.db_total_support
            )
            bv[i] = prev_i

        return next_state_support

    def _get_reward(self, bv):
        reward = 0
        # If bit-vector is UPBV, then minus reward
        if not self._check_pbv(tuple(bv)):
            self.total_reward += self.minus_reward
            return self.minus_reward
        # If the bit-vector is an unextracted FI, a large positive reward
        if self._record_new_fp(bv):
            reward += self.plus_reward
        # If bit-vector is promising, reward it accordingly
        reward += self._get_support_reward(bv)
        self.total_reward += reward
        return reward

    def _get_support_reward(self, bv):
        sup = self._calc_support(tuple(bv)).support
        if self.min_sup <= sup:
            return 4
        elif 3 / 4 * self.min_sup <= sup:
            return 3
        elif 1 / 2 * self.min_sup <= sup:
            return 2
        elif 1 / 4 * self.min_sup <= sup:
            return 1
        else:
            return 0

    def _observe(self):
        return self.get_next_state_support(self.bit_vector)

    def _is_done(self):
        return self.steps >= self.max_steps

    def _create_random_pbv(self):
        for t in count():
            random_pbv = np.random.binomial(1, self.prob_bit_stand)
            if self._check_pbv(tuple(random_pbv)):
                self.num_random_bv += t + 1
                return random_pbv
            if t == 1e5:
                print("Too many random searches")
                sys.exit()

    def _convert_bv2tuple_x(self, bv):
        return tuple(sorted([self.b2i_dict[i] for i, v in enumerate(bv) if v == 1]))

    @lru_cache(maxsize=LIMIT)
    def _check_pbv(self, bv):
        # True if PBV, False otherwise (UPBV)
        bv = np.array(bv, dtype=np.int8)
        bv_mask = bv > 0
        filtered = self.bit_map[:, bv_mask]
        if filtered.size == 0:
            return False
        if np.any(np.all(filtered, axis=1)):
            return True
        return False

    def _record_new_fp(self, bv):
        # True if not extracted, False otherwise
        pattern = self._calc_support(tuple(bv))
        if (pattern.support >= self.min_sup) and (pattern not in self.sfp):
            self.sfp.add(pattern)
            return True
        else:
            return False

    @lru_cache(maxsize=LIMIT)
    def _calc_support(self, bv):
        bv = np.array(bv, dtype=np.int8)
        itemsets = self._convert_bv2tuple_x(bv)
        if not self._is_support_gt_quarter(bv):
            return Pattern(itemsets, 0)

        bv_mask = bv > 0
        # After masking the bit-vector, check if all the elements of the itemset are present.
        # masking the bit-vector
        filtered = self.bit_map[:, bv_mask]
        if filtered.size == 0:
            return Pattern(itemsets, 0)
        # Extract columns where all elements of the itemset are zero or greater and calculate the sum.
        support = np.sum(np.all(filtered, axis=1))
        return Pattern(itemsets, support)

    def _is_support_gt_quarter(self, bv):
        return np.sum(self.each_htwfp_support * bv) >= 1 / 4 * self.min_sup
