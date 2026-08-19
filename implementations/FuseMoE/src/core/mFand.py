'''
mFand module
'''

import torch
import math
from torch import nn
import torch.nn.functional as F


class MultiFeatureAttention(nn.Module):
    def __init__(self, embed_value=32, num_heads=8, input_value_dim=32, input_dim=32, nhidden=32, dropout=0.1):
        super(MultiFeatureAttention, self).__init__()

        # value emb
        self.learn_value_embedding = nn.Sequential(
            nn.Linear(input_value_dim, nhidden),
            nn.Tanh(),
            nn.Linear(nhidden, embed_value, bias=False),
        )

        assert embed_value % num_heads == 0
        self.embed_value = embed_value
        self.embed_value_k = embed_value // num_heads
        self.h = num_heads
        self.nhidden = nhidden  # output dim
        self.linears = nn.ModuleList([nn.Linear(embed_value, embed_value),
                                      nn.Linear(embed_value, embed_value),
                                      nn.Linear(input_dim * num_heads, nhidden)])

        self.dropout=dropout

        # self.query_data_enc = nn.Sequential(
        #     nn.Linear(int(input_value_dim/2), nhidden),
        #     nn.LayerNorm(nhidden),
        #     nn.ReLU(),
        #     nn.Linear(nhidden, nhidden),
        # )

    def attention(self, query, key, value, mask=None):
        "Compute 'Scaled Dot Product Attention'"

        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)  # B H L_t L
        # print(scores.shape)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)  # B H L_t L dim  最后一维复制
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(-1)  # B 1 L 1

            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim=-2)  # B H L_t L dim  在L维度上归一化
        if self.dropout is not None:
            p_attn = F.dropout(p_attn, p=self.dropout, training=self.training)

        return torch.sum(p_attn * value.unsqueeze(1), -2), p_attn  # B H L_t L dim * B 1 1 L dim; 然后在L维上求和-> B H L_t dim

    def forward(self, query_value, key_value, value, emb_mask=None, impute_data=None):
        # query tt: B L_t D
        # key tt: B L D
        # value: B L D
        # emb mask: B L

        value = value.unsqueeze(1)  # B 1 L D

        # 1. tt->value emb
        query_value_emb = self.learn_value_embedding(query_value)
        key_value_emb = self.learn_value_embedding(key_value)
        value_emb = self.learn_value_embedding(value)

        # 2. value emb->linear->split to n head
        batch, _, seq_len, dim = value_emb.shape
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_value_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query_value_emb, key_value_emb))]  # B H L E/D

        # 3. attention mask  B L -> B 1 L
        if emb_mask is not None:
            # Same mask applied to all h heads.
            emb_mask = emb_mask.unsqueeze(1)  # B 1 L

        # 4. attention
        x, _ = self.attention(query, key, value_emb, emb_mask)
        x = x.transpose(1, 2).contiguous() \
            .view(batch, -1, self.h * dim)

        # 5. query value
        # query_data = self.query_data_enc(impute_data)
        return self.linears[-1](x)