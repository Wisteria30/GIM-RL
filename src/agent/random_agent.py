# -*- coding: utf-8 -*-

import random
import torch

from src.agent import Agent


class RandomAgent(Agent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)

    def select_action(self, state):
        return torch.tensor(
            [[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long
        )
