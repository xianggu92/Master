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
        query = rearrange(query, 'b (n_patch patch_len) hidden_dim -> (b n_patch) patch_len hidden_dim', n_patch=self.n_patch)
        key, value, mask, x_time = self.patchify_all(key, value, mask, x_time)
    
        # 取得局部特徵
        output = self.local_encoder(query, key, value, mask)
        output = self.unpatchify(output)

        # 融合全局與局部特徵
        if self.use_global:
            score = torch.sigmoid(self.score_layer(global_output))
            output = output + score * global_output

        return output

    def patchify_all(self, key, value, mask, x_time):
        """
        將 key, value, mask, x_time 根據時間點分配到對應的 patch，並 pad 到該 batch 中的最大長度。
        """
        B, L, D = key.shape
        device = key.device

        # 將文字模態的 Mask 新增變數維度
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(-1)

        # 建立有效 Mask 避免 Padding 被考慮成有實際觀測值的時間點
        valid_mask = ((1 - mask).sum(dim=-1) > 0) # (B, L)

        # 計算每個時間點屬於哪一個 patch ID (範圍：0 ~ n_patch-1)
        patch_width = 1.0 / self.n_patch
        patch_ids = torch.clamp((x_time / patch_width).long(), 0, self.n_patch - 1) # (B, L)

        # 如果是 padding ，給它一個無效的 id
        patch_ids = torch.where(valid_mask, patch_ids, torch.tensor(-1, device=device))

        # 為每個 (batch_idx, patch_idx) 統計裡面有多少個點
        counts = torch.zeros((B, self.n_patch), dtype=torch.long, device=device)
        
        # 為了做向量化索引，我們需要知道每個點在它所屬 patch 內部的相對位置 (Offset)
        offsets = torch.zeros_like(patch_ids)
        for b in range(B):
            for l in range(L):
                p_id = patch_ids[b, l]

                # 只有有效點才統計
                if p_id != -1:
                    offsets[b, l] = counts[b, p_id]
                    counts[b, p_id] += 1

        # 找出當前 Batch 內所有 Patch 中最大長度
        max_len = counts.max().item()
        
        # 初始化目標張量，預設填入 Padding 值
        new_key = torch.zeros((B * self.n_patch, max_len, D), device=device)
        new_value = torch.zeros((B * self.n_patch, max_len, value.shape[-1]), device=device)
        new_mask = torch.ones((B * self.n_patch, max_len, mask.shape[-1]), device=device, dtype=mask.dtype) # 預設全遮
        new_time = torch.zeros((B * self.n_patch, max_len, 1), device=device)

        # 計算展平後的新座標索引： (b * n_patch + p_id) * max_len + offset
        # 計算每個 Batch 在展平後是從第幾個 Patch 開始，並將每個時間點的 Patch ID 進行位移
        batch_offsets = torch.arange(B, device=device).unsqueeze(-1) * self.n_patch # (B, 1)
        flat_patch_indices = patch_ids + batch_offsets # (B, L)
        
        # 將單位從 Patch 轉換成時間點後，加上每個 Patch 內部的位移
        final_indices = flat_patch_indices * max_len + offsets # (B, L)

        # 過濾出有效點的索引
        valid_flat_mask = valid_mask.view(-1)
        final_indices = final_indices.view(-1)[valid_flat_mask]

        # 展平原始資料，利用索引直接將資料填入對應的位置
        new_key.view(-1, D).index_copy_(0, final_indices, key.view(-1, D)[valid_flat_mask])
        new_value.view(-1, value.shape[-1]).index_copy_(0, final_indices, value.view(-1, value.shape[-1])[valid_flat_mask])
        new_mask.view(-1, mask.shape[-1]).index_copy_(0, final_indices, mask.view(-1, mask.shape[-1])[valid_flat_mask])
        new_time.view(-1, 1).index_copy_(0, final_indices, x_time.view(-1, 1)[valid_flat_mask])

        return new_key, new_value, new_mask, new_time
    
    def unpatchify(self, x):
        return rearrange(x, "(b n_patch) patch_len hidden_dim -> b (n_patch patch_len) hidden_dim", n_patch=self.n_patch)