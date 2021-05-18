# -*- coding: utf-8 -*-

import hydra
from itertools import count
from omegaconf import DictConfig
import os
import torch
from tqdm import tqdm

from src.env import AssociationRuleMining
from src.mlflow_writer import MlflowWriter
from src.model import ReplayMemory
from src.preprocessing import set_multi_agent, set_common_tag, set_seed


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    original_cwd = hydra.utils.get_original_cwd()
    experiment_name = cfg.experiments
    tracking_uri = os.path.join(original_cwd, "mlruns/")
    writer = MlflowWriter(experiment_name, tracking_uri)
    writer.log_params_from_omegaconf_dict(cfg)

    # tag settting
    set_common_tag(writer, cfg)
    # seed setting
    set_seed(cfg.seed)

    env = AssociationRuleMining(
        sup_delta=cfg.dataset.ar.sup_delta,
        conf_delta=cfg.dataset.ar.conf_delta,
        data_path=os.path.join(original_cwd, cfg.dataset.ar.file_path),
        used=cfg.dataset.ar.used,
        head=cfg.dataset.ar.head,
        shuffle_db=cfg.dataset.ar.shuffle_db,
        plus_reward=cfg.env.plus_reward,
        minus_reward=cfg.env.minus_reward,
        max_steps=cfg.env.max_steps,
        cache_limit=cfg.env.cache_limit,
    )
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    target_update = cfg.interaction.target_update
    episodes = cfg.interaction.episodes
    agent = set_multi_agent(env, cfg)
    memory = ReplayMemory(cfg.interaction.replaymemory_size)
    total_ar = set()
    loss = None

    # Training
    with tqdm(range(episodes)) as pbar:
        for i in pbar:
            pbar.set_description("[Episode %d]" % i)
            state = torch.from_numpy(env.reset()).to(dtype=torch.float, device=device)
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action.item())
                next_state = torch.from_numpy(next_state).to(
                    dtype=torch.float, device=device
                )
                reward = torch.tensor([reward], device=device)
                memory.push(state, action, next_state, reward)
                state = next_state
                loss = agent.optimize_agent(memory)
                if done:
                    break
            total_ar = total_ar | env.sar
            writer.log_metric("Total Discovered AR", len(total_ar), step=i)
            writer.log_metric("Discovered AR", len(env.sar), step=i)
            writer.log_metric("counting random bitvector", env.num_random_bv, step=i)
            writer.log_metric("Reward", env.total_reward, step=i)
            if loss:
                writer.log_metric("Loss", loss.item(), step=i)
            if i % target_update == 0:
                agent.update_agent()
    # Reinsert found itemsets into env to format them
    env.sar = total_ar
    env.render()
    model = agent.get_model()

    if model is not None:
        writer.log_torch_model(model)
    writer.set_terminated()
    writer.post_slack("ar", os.path.join(os.getcwd(), ".hydra/overrides.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/config.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/hydra.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), ".hydra/overrides.yaml"))
    writer.log_artifact(os.path.join(os.getcwd(), "ar_train.log"))
    writer.log_artifact(os.path.join(os.getcwd(), "result.txt"))


if __name__ == "__main__":
    main()
