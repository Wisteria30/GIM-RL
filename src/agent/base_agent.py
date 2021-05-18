# -*- coding: utf-8 -*-

import torch


class Agent:
    def __init__(self, env, cfg):
        # if gpu is to be used
        self.device = torch.device(
            f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
        )
        self.env = env
        self.cfg = cfg
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

    def select_action(self, state):
        pass

    def optimize_agent(self, memory):
        pass

    def update_agent(self):
        pass

    def get_model(self):
        return None

    def reset_agent4test(self, **kwargs):
        pass
