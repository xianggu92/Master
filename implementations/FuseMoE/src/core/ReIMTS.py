# Code from: https://github.com/Ladbaby/PyOmniTS
import importlib
import math
import subprocess
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from einops import rearrange, repeat
from torch import Tensor
from core.module import multiTimeAttention
from typing import Optional


class ReIMTS(nn.Module):
    '''
    - paper: "Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting" (ICLR 2026)
    - paper link: https://openreview.net/forum?id=JEIDxiTWzB
    - code adapted from: https://github.com/Ladbaby/PyOmniTS
    '''
    def __init__(
        self,
        input_dim,
        hidden_dim,
        embed_time,
        num_heads,
        tt_max,
        current_level: int = 0
    ) -> None:
        super().__init__()
        self.current_level = current_level
        self.time_len_list = [tt_max, tt_max // 2]
        self.n_levels = len(self.time_len_list)
    
        self.current_time_len = self.time_len_list[self.current_level]
        if self.current_level == 0:
            self.n_patch_all = 1
        else:
            self.n_patch_all = math.ceil(self.time_len_list[self.current_level - 1] / self.time_len_list[self.current_level])

        # dynamic backbone model import & construction
        self.backbone = multiTimeAttention(input_dim, hidden_dim, embed_time, num_heads, reimts=True, num_ref_points=self.current_time_len)

        if self.current_level < self.n_levels - 1:
            # recursively creates model in each scale level
            self.next_model = ReIMTS(
                input_dim,
                hidden_dim,
                embed_time,
                num_heads,
                tt_max,
                current_level=current_level + 1
            )

    def forward(
        self, 
        query: Tensor, 
        key: Optional[Tensor] = None, 
        value: Optional[Tensor] = None, 
        x_time: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        x_repr_time: Optional[Tensor] = None,
    ):
        input_dict = {
            "query": query,
            "key": key,
            "value": value,
            "x_time": x_time,
            "mask": mask,
            "x_repr_time": x_repr_time,
        }

        backbone_output = self.backbone(**input_dict)

        if self.current_level == self.n_levels - 1:
            # base case in lowest layer, recursion stops.
            return backbone_output
        else:
            N_PATCH_NEXT = math.ceil(self.time_len_list[self.current_level] / self.time_len_list[self.current_level + 1])
            query = rearrange(query, 'B (P PATCH_LEN) D -> (B P) PATCH_LEN D', P=N_PATCH_NEXT)
            key = self.patchify(key)
            value = self.patchify(value)
            mask = self.patchify(mask, x_time)
            x_time = self.patchify(x_time)

            input_dict = {
                "query": query,
                "key": key,
                "value": value,
                "x_time": x_time,
                "mask": mask,
            }

            input_dict["x_repr_time"] = self.patchify_repr_time(backbone_output)

            next_model_output = self.next_model(**input_dict) # recursively invoke model in next scale level
            next_model_output = self.unpatchify(next_model_output)

            return next_model_output

    def patchify(self, x: Tensor, x_time: Tensor = None):
        N_PATCH_NEXT = math.ceil(self.time_len_list[self.current_level] / self.time_len_list[self.current_level + 1])

        if x_time is None:
            return x.repeat_interleave(N_PATCH_NEXT, dim=0)
        
        # 2. 複製 mask 與時間點 (Total: B * L * n_patch)
        # 這裡 dim=0 會讓順序變成 [P1_data, P2_data, ...]
        x_expanded = x.repeat_interleave(N_PATCH_NEXT, dim=0)
        x_time_expanded = x_time.repeat_interleave(N_PATCH_NEXT, dim=0)
        
        # 3. 建立 Patch 索引矩陣
        # 產生如 [0, 1, 2, ..., n_patch-1, 0, 1, 2, ...] 的序列
        # 需要與 x_expanded 的形狀對齊
        device = x.device
        num_elements = x.shape[0]
        patch_indices = torch.arange(N_PATCH_NEXT, device=device).repeat(num_elements)
        
        # 4. 計算每個 Patch 的邊界
        patch_width = 1.0 / N_PATCH_NEXT
        patch_start = patch_indices * patch_width
        patch_end = patch_start + patch_width
        
        # 5. 判斷時間點是否在區間外 (不在區間內則設為 True)
        # x_time_expanded 需與 patch_start 比較
        # print(x_time_expanded.shape) # [4, 106]
        # print(patch_start.shape) # [4]
        out_of_patch = (x_time_expanded < patch_start.unsqueeze(-1)) | (x_time_expanded >= patch_end.unsqueeze(-1))
        
        # 6. 將 Patch 外的 mask 設為 1
        # 注意：這裡假設 x 的 mask 1 是代表「無效/遮蔽」，0 是「有效」
        # 如果你的邏輯相反，請調整判斷式
        x_expanded[out_of_patch] = 1.0
        
        return x_expanded

    def patchify_repr_time(self, x):
        N_PATCH_NEXT = math.ceil(self.time_len_list[self.current_level] / self.time_len_list[self.current_level + 1])
        return rearrange(x, 'b (n p) d -> (b n) p d', n=N_PATCH_NEXT)
    
    def unpatchify(self, x):
        N_PATCH_ALL_NEXT_FRACTAL = math.ceil(self.time_len_list[self.current_level] / self.time_len_list[self.current_level + 1])
        return rearrange(x, "(B N_PATCH) PATCH_LEN ENC_IN -> B (N_PATCH PATCH_LEN) ENC_IN", N_PATCH=N_PATCH_ALL_NEXT_FRACTAL)