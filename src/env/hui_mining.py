# -*- coding: utf-8 -*-

from collections import namedtuple
from functools import lru_cache
import gym
import gym.spaces
from itertools import count
import numpy as np
import sys

from src.env.db_loader import load_hui_db


LIMIT = 100000
Rule = namedtuple("Rule", ("itemsets", "utility"))


class HighUtilityItemsetsMining(gym.Env):
    def __init__(
        self,
        delta=0.3,
        data_path="data/chess_utility_spmf.txt",
        used=1.0,
        head=0.0,
        shuffle_db=False,
        plus_reward=1,
        minus_reward=-1,
        max_steps=1000,
        cache_limit=100000,
    ):
        super().__init__()
        self.database, self.items, self.utils = load_hui_db(
            data_path, used, head, shuffle_db
        )
        # Specify the maximum LRU cache as a global variable
        global LIMIT
        LIMIT = cache_limit
        # Setting minimum utility threshold
        self.delta = delta
        self.db_total_util = np.sum(self.utils)
        self.min_util = self.delta * self.db_total_util
        self._setup_db()
        self.max_steps = max_steps
        # Action = invert each bit + regenerate bit vector
        self.action_space = gym.spaces.Discrete(len(self.htwui_1) + 1)
        # Observation = each utility
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=((len(self.htwui_1)),)
        )
        self.plus_reward = plus_reward
        self.minus_reward = minus_reward
        self.reward_range = [float(minus_reward), float(plus_reward)]
        self.done = False
        self.steps = 0
        self.num_random_bv = 0
        self.total_reward = 0
        # Calculate the utility of each item
        self.each_htwui_utility = np.sum(self.bit_map, axis=0)
        self.reset()

    def _setup_db(self):
        # high transaction-weighted utilization itemset
        self.htwui_1 = set()
        for item in self.items:
            twu = 0
            for i, transaction in enumerate(self.database):
                if item in transaction:
                    twu += self.utils[i]
            if twu >= self.min_util:
                self.htwui_1.add(item)

        self.b2i_dict = {i: v for i, v in enumerate(self.htwui_1)}
        self.i2b_dict = {v: i for i, v in enumerate(self.htwui_1)}

        self.bit_map = np.zeros((len(self.database), len(self.htwui_1)), dtype=np.int)
        self.bit_vector = None  # bv
        for i, transaction in enumerate(self.database):
            for item in transaction:
                if item in self.htwui_1:
                    self.bit_map[i][self.i2b_dict[item]] = self.database[i][
                        item
                    ].astype(np.int)

        # Probability that each item appears on the DB
        self.prob_bit_stand = np.sum(self.bit_map > 0, axis=0) / len(self.database)
        # Set of High Utility Itemset
        self.shui = set()

    def reset(self):
        if self.bit_vector is None or not self._check_pbv(tuple(self.bit_vector)):
            self.bit_vector = self._create_random_pbv()

        self.done = False
        self.steps = 0
        self.shui = set()
        self.num_random_bv = 0
        return self._observe()

    def step(self, action):
        # The last index of the action is the regeneration of the bit-vector
        if action == len(self.htwui_1):
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
        for hui in self.shui:
            itemsets = " ".join(map(lambda x: str(int(x)), hui.itemsets))
            result.append("{} #UTIL: {}".format(itemsets, hui.utility))
        with open("result.txt", mode="w") as f:
            f.write("\n".join(result))

    def get_next_state_utility(self, bv):
        next_state_utility = np.zeros(len(self.htwui_1))
        for i in range(len(self.htwui_1)):
            # Invert each bit and check if it is a PBV.
            # Reduce the amount of computation by changing the bit-vector directly.
            prev_i = bv[i]
            bv[i] = 1 if bv[i] == 0 else 0
            if not self._check_pbv(tuple(bv)):
                bv[i] = prev_i
                continue
            rule = self._calc_utility(tuple(bv))
            next_state_utility[i] = np.log1p(rule.utility) / np.log1p(
                self.db_total_util
            )
            bv[i] = prev_i

        return next_state_utility

    def _get_reward(self, bv):
        reward = 0
        # If bit-vector is UPBV, then minus reward
        if not self._check_pbv(tuple(bv)):
            self.total_reward += self.minus_reward
            return self.minus_reward
        # If the bit-vector is an unextracted HUI, a large positive reward
        if self._record_new_rule(bv):
            reward += self.plus_reward
        # If bit-vector is promising, reward it accordingly
        reward += self._get_utility_reward(bv)
        self.total_reward += reward
        return reward

    def _get_utility_reward(self, bv):
        util = self._calc_utility(tuple(bv)).utility
        if self.min_util <= util:
            return 4
        elif 3 / 4 * self.min_util <= util:
            return 3
        elif 1 / 2 * self.min_util <= util:
            return 2
        elif 1 / 4 * self.min_util <= util:
            return 1
        else:
            return 0

    def _observe(self):
        return self.get_next_state_utility(self.bit_vector)

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

    def _record_new_rule(self, bv):
        # True if not extracted, False otherwise
        rule = self._calc_utility(tuple(bv))
        if (rule.utility >= self.min_util) and (rule not in self.shui):
            self.shui.add(rule)
            return True
        else:
            return False

    @lru_cache(maxsize=LIMIT)
    def _calc_utility(self, bv):
        bv = np.array(bv, dtype=np.int8)
        itemsets = self._convert_bv2tuple_x(bv)
        if not self._is_utility_gt_quarter(bv):
            return Rule(itemsets, 0)

        bv_mask = bv > 0
        # After masking the bit-vector, check if all the elements of the itemset are present.
        # masking the bit-vector
        filtered = self.bit_map[:, bv_mask]
        if filtered.size == 0:
            return Rule(itemsets, 0)
        # Extract columns where all elements of the itemset are zero or greater and calculate the sum.
        utility = np.sum(filtered[np.all(filtered, axis=1)])
        return Rule(itemsets, utility)

    def _is_utility_gt_quarter(self, bv):
        return np.sum(self.each_htwui_utility * bv) >= 1 / 4 * self.min_util
