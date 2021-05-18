# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiInputModule(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.half_dim = input_dim // 2
        self.sup_bn = nn.BatchNorm1d(num_features=self.half_dim)
        self.conf_bn = nn.BatchNorm1d(num_features=self.half_dim)

    def forward(self, x):
        sup = torch.narrow(x, 1, 0, self.half_dim)
        conf = torch.narrow(x, 1, self.half_dim, self.half_dim)
        sup = self.sup_bn(sup)
        conf = self.conf_bn(conf)
        x = torch.cat([sup, conf], dim=1)
        return x


class MultiQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, cfg):
        super().__init__()

        self.multi_input = MultiInputModule(input_dim=input_dim)
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
        x = self.multi_input(x)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        x = F.leaky_relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


class MultiSimpleNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        hidden_1 = 4096

        self.multi_input = MultiInputModule(input_dim=input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.fc2 = nn.Linear(hidden_1, output_dim)

    def forward(self, x):
        x = self.multi_input(x)
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        return self.fc2(x)
