# -*- coding: utf-8 -*-

import torch

from src.agent import Agent


class MultiMaxAgent(Agent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)

    def select_action(self, state):
        support_state = torch.split(state, state.size()[0] // 2, dim=0)[0]
        return torch.tensor(
            [[support_state.argmax()]], device=self.device, dtype=torch.long
        )
