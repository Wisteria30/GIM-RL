# -*- coding: utf-8 -*-

import numpy as np
import random
import os
import sys
import torch

from src.agent import (
    EpsilonGreedyAgent,
    MaxAgent,
    RandomAgent,
    RandomCreateBVAgent,
    ProbabilityAgent,
    QAgent,
    QAndUtilityAgent,
    MultiEpsilonGreedyAgent,
    MultiMaxAgent,
    MultiProbabilityAgent,
    MultiQAgent,
    MultiQAndUtilityAgent,
)


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def set_agent(env, cfg):
    if cfg.agent.name == "q":
        agent = QAgent(env, cfg)
    elif cfg.agent.name == "qandutility":
        agent = QAndUtilityAgent(env, cfg)
    elif cfg.agent.name == "max":
        agent = MaxAgent(env, cfg)
    elif cfg.agent.name == "probability":
        agent = ProbabilityAgent(env, cfg)
    elif cfg.agent.name == "random":
        agent = RandomAgent(env, cfg)
    elif cfg.agent.name == "randombv":
        agent = RandomCreateBVAgent(env, cfg)
    elif cfg.agent.name == "egreedy":
        agent = EpsilonGreedyAgent(env, cfg)
    else:
        print("illegal agent name.")
        sys.exit(1)
    return agent


def set_multi_agent(env, cfg):
    if cfg.agent.name == "q":
        agent = MultiQAgent(env, cfg)
    elif cfg.agent.name == "qandutility":
        agent = MultiQAndUtilityAgent(env, cfg)
    elif cfg.agent.name == "max":
        agent = MultiMaxAgent(env, cfg)
    elif cfg.agent.name == "probability":
        agent = MultiProbabilityAgent(env, cfg)
    elif cfg.agent.name == "random":
        agent = RandomAgent(env, cfg)
    elif cfg.agent.name == "randombv":
        agent = RandomCreateBVAgent(env, cfg)
    elif cfg.agent.name == "egreedy":
        agent = MultiEpsilonGreedyAgent(env, cfg)
    else:
        print("illegal agent name.")
        sys.exit(1)
    return agent


def set_common_tag(writer, cfg):
    writer.set_runname(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))
    writer.set_tag("mlflow.note.content", cfg.content)
    writer.set_tag("mlflow.user", cfg.user)
    writer.set_tag("mlflow.source.name", os.path.abspath(__file__))
    writer.set_tag("mlflow.source.git.commit", cfg.commit)
