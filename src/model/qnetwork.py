# -*- coding: utf-8 -*-

from collections import namedtuple
import random
import torch.nn as nn
import torch.nn.functional as F


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def reset(self):
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, cfg):
        super().__init__()

        self.fc1 = nn.Linear(in_features=input_dim, out_features=cfg.model.hidden_1)
        self.bn1 = nn.BatchNorm1d(num_features=cfg.model.hidden_1)
        self.fc2 = nn.Linear(
            in_features=cfg.model.hidden_1, out_features=cfg.model.hidden_2
        )
        self.bn2 = nn.BatchNorm1d(num_features=cfg.model.hidden_2)
        self.fc3 = nn.Linear(
            in_features=cfg.model.hidden_2, out_features=cfg.model.hidden_3
        )
        self.bn3 = nn.BatchNorm1d(num_features=cfg.model.hidden_3)
        self.fc4 = nn.Linear(in_features=cfg.model.hidden_3, out_features=output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


class SimpleNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_1 = 4096

        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.fc2 = nn.Linear(hidden_1, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        return self.fc2(x)
