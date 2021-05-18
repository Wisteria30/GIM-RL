# -*- coding: utf-8 -*-

import torch

from src.agent import Agent


class RandomCreateBVAgent(Agent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)

    def select_action(self, state):
        return torch.tensor(
            [[self.n_actions - 1]], device=self.device, dtype=torch.long
        )
