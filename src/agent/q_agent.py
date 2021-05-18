# -*- coding: utf-8 -*-

import math
import random
import torch
from torch import optim
import torch.nn.functional as F
import sys

from src.agent import Agent
from src.model import QNetwork, Transition, RAdam, SimpleNetwork


class QAgent(Agent):
    def __init__(self, env, cfg):
        super().__init__(env, cfg)
        self.batch_size = cfg.agent.batch_size
        self.gamma = cfg.agent.gamma
        self.eps_start = cfg.agent.eps_start
        self.eps_end = cfg.agent.eps_end
        self.eps_decay = cfg.agent.eps_decay
        self.policy_net, self.target_net = map(
            lambda x: x.to(self.device),
            self.set_network(self.n_states, self.n_actions, cfg),
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.steps_done = 0
        self.optimizer = self.set_optimizer(cfg)

    def set_optimizer(self, cfg):
        if cfg.agent.optimizer == "sgd":
            optimizer = optim.SGD(self.policy_net.parameters(), lr=0.01)
        elif cfg.agent.optimizer == "rmsprop":
            optimizer = optim.RMSprop(self.policy_net.parameters())
        elif cfg.agent.optimizer == "adam":
            optimizer = optim.Adam(self.policy_net.parameters())
        elif cfg.agent.optimizer == "radam":
            optimizer = RAdam(self.policy_net.parameters())
        else:
            print("illegal optimizer")
            sys.exit(1)
        return optimizer

    def set_network(self, input_dim, output_dim, cfg):
        if cfg.agent.network == "default":
            policy_net = QNetwork(input_dim, output_dim, cfg)
            target_net = QNetwork(input_dim, output_dim, cfg)
        elif cfg.agent.network == "simple":
            policy_net = SimpleNetwork(input_dim, output_dim)
            target_net = SimpleNetwork(input_dim, output_dim)
        return (policy_net, target_net)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                self.policy_net.eval()
                q_value = self.policy_net(state.view(1, -1)).to("cpu")
                self.policy_net.train()
                return q_value.max(1)[1].view(1, 1).to(device=self.device)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=self.device,
                dtype=torch.long,
            )

    def optimize_agent(self, memory):
        if len(memory) < self.batch_size:
            return
        transitions = memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.stack(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.stack(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = (
            self.target_net(non_final_next_states).max(1)[0].detach()
        )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def update_agent(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_model(self):
        return self.policy_net.cpu()

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

    def reset_network(self, cfg, model, shared_weights_column, test):
        if cfg.agent.network == "default":
            if test != "random":
                # Substitute all layers except fc1 and fc4 as they are
                self.policy_net.fc2.weight.data = model.fc2.weight.data.to(self.device)
                self.policy_net.fc3.weight.data = model.fc3.weight.data.to(self.device)
                self.policy_net.bn1.weight.data = model.bn1.weight.data.to(self.device)
                self.policy_net.bn2.weight.data = model.bn2.weight.data.to(self.device)
                self.policy_net.bn3.weight.data = model.bn3.weight.data.to(self.device)
                if test != "not_fc1":
                    # Train: Test order dictionary, so v, k
                    for k, v in shared_weights_column.items():
                        self.policy_net.fc1.weight.data[:, v] = model.fc1.weight.data[
                            :, k
                        ].to(self.device)
                        self.policy_net.fc4.weight.data[v] = model.fc4.weight.data[
                            k
                        ].to(self.device)
        elif cfg.agent.network == "simple":
            if test != "random":
                # Substitute all layers except fc1 and fc4 as they are
                self.policy_net.bn1.weight.data = model.bn1.weight.data.to(self.device)
                if test != "not_fc1":
                    # Train: Test order dictionary, so v, k
                    for k, v in shared_weights_column.items():
                        self.policy_net.fc1.weight.data[:, v] = model.fc1.weight.data[
                            :, k
                        ].to(self.device)
                        self.policy_net.fc2.weight.data[v] = model.fc2.weight.data[
                            k
                        ].to(self.device)
