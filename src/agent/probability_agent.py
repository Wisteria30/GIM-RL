# -*- coding: utf-8 -*-

import torch
import random

from src.agent import Agent


class ProbabilityAgent(Agent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)

    def select_action(self, state):
        return torch.tensor(
            [random.choices(range(self.n_states), weights=state.to("cpu"))],
            device=self.device,
            dtype=torch.long,
        )
