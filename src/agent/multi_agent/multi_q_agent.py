# -*- coding: utf-8 -*-

from torch import optim
import sys

from src.agent import QAgent
from src.model import MultiQNetwork, MultiSimpleNetwork


class MultiQAgent(QAgent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.policy_net, self.target_net = map(
            lambda x: x.to(self.device),
            self.set_network(self.n_states, self.n_actions, cfg),
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = self.set_optimizer(cfg)

    def set_network(self, input_dim, output_dim, cfg):
        if cfg.agent.network == "default":
            policy_net = MultiQNetwork(input_dim, output_dim, cfg)
            target_net = MultiQNetwork(input_dim, output_dim, cfg)
        elif cfg.agent.network == "simple":
            policy_net = MultiSimpleNetwork(input_dim, output_dim)
            target_net = MultiSimpleNetwork(input_dim, output_dim)
        return (policy_net, target_net)

    def reset_agent4test(self, **kwargs):
        model = kwargs["model"]
        shared_weights_column = kwargs["shared_weights_column"]
        test = kwargs["test"]
        self.policy_net, self.target_net = map(
            lambda x: x.to(self.device),
            self.set_network(self.n_states, self.n_actions, self.cfg),
        )
        self.reset_network(self.cfg, model, shared_weights_column, test)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = self.set_optimizer(self.cfg)
        self.eps_start = self.cfg.agent.test_eps_start
        self.eps_end = self.cfg.agent.test_eps_end
        self.eps_decay = self.cfg.agent.test_eps_decay
        self.steps_done = 0
