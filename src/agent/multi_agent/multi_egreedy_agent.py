# -*- coding: utf-8 -*

import math
import random
import torch

from src.agent import Agent


class MultiEpsilonGreedyAgent(Agent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.eps_start = cfg.agent.eps_start
        self.eps_end = cfg.agent.eps_end
        self.eps_decay = cfg.agent.eps_decay
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        support_state = torch.split(state, state.size()[0] // 2, dim=0)[0]
        if sample > eps_threshold:
            return torch.tensor(
                [[support_state.argmax()]], device=self.device, dtype=torch.long
            )
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

    def reset_agent4test(self, **kwargs):
        self.eps_start = self.cfg.agent.test_eps_start
        self.eps_end = self.cfg.agent.test_eps_end
        self.eps_decay = self.cfg.agent.test_eps_decay
        self.steps_done = 0
