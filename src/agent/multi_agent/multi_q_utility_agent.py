# -*- coding: utf-8 -*-

import math
import random
import torch

from src.agent import MultiQAgent


class MultiQAndUtilityAgent(MultiQAgent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.lambda_start = cfg.agent.lambda_start
        self.lambda_end = cfg.agent.lambda_end
        self.lambda_decay = cfg.agent.lambda_decay

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        lambda_threshold = self.lambda_end + (
            self.lambda_start - self.lambda_end
        ) * math.exp(-1.0 * self.steps_done / self.lambda_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                utility = self.utility_probability(
                    state, self.cfg.agent.random_utility_per
                )
                self.policy_net.eval()
                q_value = self.policy_net(state.view(1, -1)).to("cpu")
                self.policy_net.train()
                q_value = q_value / q_value.sum()
                action_value = (
                    lambda_threshold * utility + (1 - lambda_threshold) * q_value
                )
                return action_value.max(1)[1].view(1, 1).to(device=self.device)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

    # Probabilistically compute the itemset of the next state candidate. 
    # For random actions, the next state is not known, so specify an arbitrary probability.
    # Note that the array of returned values is one length longer than the array of arguments.
    def utility_probability(self, x, random_per=0.02):
        x = x.to("cpu")
        x = torch.split(x, x.size()[0] // 2, dim=0)[0]
        random_x = torch.Tensor([(random_per / (1 - random_per)) * x.sum()])
        xx = torch.cat((x, random_x), 0)
        return xx / xx.sum()

    def reset_agent4test(self, **kwargs):
        super().reset_agent4test(**kwargs)
        self.lambda_start = self.cfg.agent.test_eps_start
        self.lambda_end = self.cfg.agent.test_eps_end
        self.lambda_decay = self.cfg.agent.test_eps_decay
