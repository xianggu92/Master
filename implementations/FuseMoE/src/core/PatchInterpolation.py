import torch
import torch.nn as nn
from einops import rearrange
from core.module import multiTimeAttention


class PatchInterpolation(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_time, num_heads, tt_max, n_patch, n_ref_point, use_global=False):
        super().__init__()
        self.tt_max = tt_max
        self.n_patch = n_patch
        self.n_ref_point = n_ref_point
        self.use_global = use_global

        # Reference points 數量必須能夠被 Patch 數量整除
        assert self.n_ref_point % self.n_patch == 0

        self.local_encoder = multiTimeAttention(input_dim, hidden_dim, embed_time, num_heads)

        if use_global:
            self.global_encoder = multiTimeAttention(input_dim, hidden_dim, embed_time, num_heads)
            self.score_layer = nn.Linear(self.nhidden, self.nhidden)

    def forward(self, query, key, value, x_time, mask):
        # 取得全局特徵
        if self.use_global:
            global_output = self.global_encoder(query, key, value, mask)

        # 分塊，Key、Value、Time 複製成 Patch 數量，並利用 Mask 來遮蓋掉 Patch 範圍外的資訊
        query = rearrange(query, 'b (p p_len) d -> (b p) p_len d', p=self.n_patch)
        key = self.patchify(key)
        value = self.patchify(value)
        mask = self.patchify(mask, x_time)
        x_time = self.patchify(x_time)
    
        # 取得局部特徵
        output = self.local_encoder(query, key, value, mask)
        output = self.unpatchify(output)

        # 融合全局與局部特徵
        if self.use_global:
            score = torch.sigmoid(self.score_layer(global_output))
            output = output + score * global_output

        return output

    def patchify(self, x, x_time=None):
        # Mask 以外的資料直接複製，利用 Mask 來遮蓋掉 Patch 外的資料
        if x_time is None:
            return x.repeat_interleave(self.n_patch, dim=0)
        
        # Batch 內的每筆 Mask 與 Time 複製 Patch 數量的次數
        x_expanded = x.repeat_interleave(self.n_patch, dim=0)
        x_time_expanded = x_time.repeat_interleave(self.n_patch, dim=0)
        
        # 產生如 [0, 1, 2, ..., n_patch-1, 0, 1, 2, ...] 的序列
        device = x.device
        num_elements = x.shape[0]
        patch_indices = torch.arange(self.n_patch, device=device).repeat(num_elements)
        
        # 計算每個 Patch 的邊界
        patch_width = 1.0 / self.n_patch
        patch_start = patch_indices * patch_width
        patch_end = patch_start + patch_width
        
        # 判斷時間點是否在區間外 (不在區間內則設為 True)
        out_of_patch = (x_time_expanded < patch_start.unsqueeze(-1)) | (x_time_expanded >= patch_end.unsqueeze(-1))
        
        # 將 Patch 外的資料遮蓋掉
        x_expanded[out_of_patch] = 1.0
        
        return x_expanded
    
    def unpatchify(self, x):
        return rearrange(x, "(B N_PATCH) PATCH_LEN ENC_IN -> B (N_PATCH PATCH_LEN) ENC_IN", N_PATCH=self.n_patch)