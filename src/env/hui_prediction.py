# -*- coding: utf-8 -*-
from functools import lru_cache

import torch
import numpy as np
from src.model import MLP
from src.env.hui_mining import HighUtilityItemsetsMining, Rule

LIMIT = 100000


class HighUtilityItemsetsPrediction(HighUtilityItemsetsMining):
    def __init__(
        self,
        delta=0.3,
        data_path="data/chess_utility_spmf.txt",
        used=1.0,
        head=0.0,
        shuffle_db=False,
        plus_reward=1,
        minus_reward=-1,
        max_steps=1000,
        cache_limit=100000,
        model_path="models/hui/mlp/chess_utility_1000000.pth",
        device="cpu"
    ):
        self.model = None
        super().__init__(
            delta=delta,
            data_path=data_path,
            used=used,
            head=head,
            shuffle_db=shuffle_db,
            plus_reward=plus_reward,
            minus_reward=minus_reward,
            max_steps=max_steps,
            cache_limit=cache_limit,
        )
        self.device = device
        self.model = self.load_models(model_path)

    def load_models(self, model_path):
        model = MLP(len(self.items)).to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    # 元のアイテムの長さ
    def _convert_x2bv(self, bv):
        x = np.zeros(len(self.items))
        for i, v in enumerate(bv):
            if v == 1:
                item = self.b2i_dict[i]
                x[int(item) - 1] = 1.
        return x

    @lru_cache(maxsize=LIMIT)
    def _calc_utility(self, bv):
        if self.model is None:
            return super()._calc_utility(bv)

        bv = np.array(bv, dtype=np.int8)
        itemsets = self._convert_bv2tuple_x(bv)
        if not self._is_utility_gt_quarter(bv):
            return Rule(itemsets, 0)
        # bvの長さとitemの長さが違うので変換(MLPのinput_dimがアイテム長)
        X = self._convert_x2bv(bv)
        X = torch.from_numpy(X).view(1, -1).to(self.device, torch.float)
        pred_utility = int(self.model(X).item())
        # 負の値は取らないようにする
        pred_utility = max(pred_utility, 0)
        return Rule(itemsets, pred_utility)
