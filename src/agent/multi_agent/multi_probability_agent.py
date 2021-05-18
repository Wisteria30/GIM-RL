# -*- coding: utf-8 -*-

import random
import torch

from src.agent import Agent


class MultiProbabilityAgent(Agent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)

    def select_action(self, state):
        support_state = torch.split(state, state.size()[0] // 2, dim=0)[0]
        return torch.tensor(
            [
                random.choices(
                    range(self.n_states // 2), weights=support_state.to("cpu")
                )
            ],
            device=self.device,
            dtype=torch.long,
        )
