# -*- coding: utf-8 -*-

import torch

from src.agent import Agent


class MaxAgent(Agent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)

    def select_action(self, state):
        return torch.tensor([[state.argmax()]], device=self.device, dtype=torch.long)
