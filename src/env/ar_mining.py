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
# Rule a => b, support(aub), confidence(support(aub) / support(a))
Rule = namedtuple("Rule", ("a", "b", "support", "confidence"))


class AssociationRuleMining(gym.Env):
    def __init__(
        self,
        sup_delta=0.3,
        conf_delta=0.3,
        data_path="data/chess.txt",
        used=1.0,
        head=0.0,
        shuffle_db=False,
        plus_reward=200,
        minus_reward=-1,
        max_steps=10000,
        cache_limit=100000,
    ):
        super().__init__()
        self.transaction, self.bit_map, self.b2i_dict, self.i2b_dict = load_db(
            data_path, used, head, shuffle_db
        )
        # Specify the maximum LRU cache as a global variable
        global LIMIT
        LIMIT = cache_limit
        # Setting minimum support threshold and minimum confidence threshold
        self.sup_delta = sup_delta
        self.conf_delta = conf_delta
        self.db_total_support = self.bit_map.shape[0]
        self.min_sup = self.sup_delta * self.db_total_support

        self._setup_db()
        self.max_steps = max_steps
        # Action = invert each bit + regenerate bit vector
        self.action_space = gym.spaces.Discrete(len(self.htwar_1) + 1)
        # Observation = each support + each confidence
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=((len(self.htwar_1) * 2),)
        )
        self.plus_reward = plus_reward
        self.minus_reward = minus_reward
        self.reward_range = [float(minus_reward), float(plus_reward)]
        self.done = False
        self.steps = 0
        self.num_random_bv = 0
        self.total_reward = 0
        self.reset()

    def _setup_db(self):
        # high transaction-weighted association rule
        self.htwar_1 = set(
            self.b2i_dict[i]
            for i in np.where(np.sum(self.bit_map, axis=0) >= self.min_sup)[0]
        )
        self.b2i_dict = {i: v for i, v in enumerate(self.htwar_1)}
        self.i2b_dict = {v: i for i, v in enumerate(self.htwar_1)}
        self.bit_map = np.zeros(
            (self.bit_map.shape[0], len(self.htwar_1)), dtype=np.int
        )
        self.bit_vector = None  # bv
        for i, t in enumerate(self.transaction):
            for item in t:
                if item in self.htwar_1:
                    self.bit_map[i][self.i2b_dict[item]] = 1

        # Probability that each item appears on the DB
        self.prob_bit_stand = np.sum(self.bit_map, axis=0) / self.bit_map.shape[0]
        # Set of Association Rule
        self.sar = set()

    def reset(self):
        if self.bit_vector is None or not self._check_pbv(tuple(self.bit_vector)):
            self.bit_vector = self._create_random_pbv()

        self.done = False
        self.steps = 0
        self.sar = set()
        self.num_random_bv = 0
        return self._observe()

    def step(self, action):
        # The last index of the action is the regeneration of the bit-vector
        if action == len(self.htwar_1):
            self.bit_vector = self._create_random_pbv()
            action = -1
        else:
            # Bit inversion
            self.bit_vector[action] = 1 if self.bit_vector[action] == 0 else 0
        self.steps += 1
        observation = self._observe()
        reward = self._get_reward(self.bit_vector, action)
        self.done = self._is_done()
        return observation, reward, self.done, {}

    def render(self):
        result = []
        for rule in self.sar:
            a = " ".join(map(lambda x: str(int(x)), rule.a))
            b = " ".join(map(lambda x: str(int(x)), rule.b))
            result.append(
                "{} ==> {} #SUP: {} #CONF: {}".format(
                    a, b, rule.support, rule.confidence
                )
            )
        with open("result.txt", mode="w") as f:
            f.write("\n".join(result))

    def get_next_sup_conf(self, bv):
        next_state_sup = np.zeros(len(self.htwar_1))
        next_state_conf = np.zeros(len(self.htwar_1))
        for i in range(len(self.htwar_1)):
            # Invert each bit and check if it is a PBV.
            # Reduce the amount of computation by changing the bit-vector directly.
            prev_i = bv[i]
            bv[i] = 1 if bv[i] == 0 else 0
            if not self._check_pbv(tuple(bv)):
                bv[i] = prev_i
                continue
            rule = self._calc_confidence(tuple(bv), i)
            next_state_sup[i] = np.log1p(self._calc_support(tuple(bv))) / np.log1p(
                self.db_total_support
            )
            next_state_conf[i] = rule.confidence
            bv[i] = prev_i

        return np.concatenate([next_state_sup, next_state_conf])

    def _get_reward(self, bv, bit_action):
        reward = 0
        # If bit-vector is UPBV, then minus reward
        if not self._check_pbv(tuple(bv)):
            self.total_reward += self.minus_reward
            return self.minus_reward
        # If the previous action is to regenerate the bit-vector, no reward
        if bit_action == -1:
            return reward
        # If the bit-vector is an unextracted AR, a large positive reward
        if self._record_new_rule(bv, bit_action):
            reward += self.plus_reward
        # If bit-vector is promising, reward it accordingly
        reward += self._get_support_reward(bv, bit_action)
        self.total_reward += reward
        return reward

    def _get_support_reward(self, bv, bit_action):
        sup = self._calc_confidence(tuple(bv), bit_action).support
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
        return self.get_next_sup_conf(self.bit_vector)

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

    def _record_new_rule(self, bv, bit_action):
        # True if not extracted, False otherwise
        rule = self._calc_confidence(tuple(bv), bit_action)
        # If A exists (rule length k >= 2), is greater equal to min_sup,
        # is greater equal to min_conf, and is not yet found, add
        if (
            (rule.support >= self.min_sup)
            and (rule.confidence >= self.conf_delta)
            and (rule not in self.sar)
        ):
            self.sar.add(rule)
            return True
        else:
            return False

    @lru_cache(maxsize=LIMIT)
    def _calc_confidence(self, bv, action_bit):
        """
        Calculating confidence
        confidence(A => B) = P(B|A) = support(A u B) / support(A)
        action bit is the index of the bit that was inverted by the action
        """
        bv = np.array(bv, dtype=np.int8)
        b_bv = np.zeros(len(self.htwar_1))
        if bv[action_bit] == 0:
            sup_a = self._calc_support(tuple(bv))
            a = self._convert_bv2tuple_x(bv)
            b_bv[action_bit] = 1
            b = self._convert_bv2tuple_x(b_bv)
            bv[action_bit] = 1
            sup_aub = self._calc_support(tuple(bv))
        else:
            sup_aub = self._calc_support(tuple(bv))
            bv[action_bit] = 0
            sup_a = self._calc_support(tuple(bv))
            a = self._convert_bv2tuple_x(bv)
            b_bv[action_bit] = 1
            b = self._convert_bv2tuple_x(b_bv)
        # Finally, undo the bit.
        bv[action_bit] = 1 if bv[action_bit] == 0 else 0
        # If a = Φ, then conf is 0
        if len(a) == 0:
            return Rule(a, b, sup_aub, 0)
        confidence = sup_aub / sup_a
        return Rule(a, b, sup_aub, confidence)

    @lru_cache(maxsize=LIMIT)
    def _calc_support(self, bv):
        bv = np.array(bv, dtype=np.int8)

        bv_mask = bv > 0
        # After masking the bit-vector, check if all the elements of the itemset are present.
        # masking the bit-vector
        filtered = self.bit_map[:, bv_mask]
        # All elements of bit-vector are 0
        if filtered.size == 0:
            return 0
        # Extract columns where all elements of the itemset are zero or greater and calculate the sum.
        support = np.sum(np.all(filtered, axis=1))
        return support
