import torch
import torch.nn as nn
from einops import rearrange


class FFNN(nn.Module):
    def __init__(self, input_dim, hid_units, output_dim):
        self.hid_units = hid_units
        self.output_dim = output_dim
        super(FFNN, self).__init__()

        self.linear = nn.Linear(input_dim, hid_units)
        self.W = nn.Linear(hid_units, output_dim, bias=False)

    def forward(self, x):
        x = self.linear(x)
        x = self.W(torch.tanh(x))
        return x


class Var_Encoder(nn.Module):
    def __init__(self, output_dim, num_type):
        super(Var_Encoder, self).__init__()
        self.output_dim = output_dim
        self.num_type = num_type
        self.encoder = nn.Linear(1, output_dim)

    def forward(self, x, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        x = rearrange(x, 'b l k -> b l k 1')
        x = self.encoder(x)
        return x * non_pad_mask


class Type_Encoder(nn.Module):
    def __init__(self, d_model, num_types):
        super(Type_Encoder, self).__init__()
        self.event_emb = nn.Embedding(num_types, d_model)

    def forward(self, event):
        event_emb = self.event_emb(event.long())
        return event_emb


class Time_Encoder(nn.Module):
    def __init__(self, embed_time, num_types):
        super(Time_Encoder, self).__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)
        self.k_map = nn.Parameter(torch.ones(1, 1, num_types, embed_time))

    def forward(self, tt, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else:  # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')

        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        out = torch.cat([out1, out2], -1)  # [B,L,1,D]
        out = torch.mul(out, self.k_map)
        return out


class Density_Encoder(nn.Module):
    def __init__(self, embed_time, num_types, hid_dim=16):
        super(Density_Encoder, self).__init__()
        self.encoder = FFNN(1, hid_dim, embed_time)
        self.k_map = nn.Parameter(torch.ones(1, 1, num_types, embed_time))

    def forward(self, tt, non_pad_mask):
        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b l k 1')
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, 'b l k -> b l k 1')
        else:  # [B,L]
            tt = rearrange(tt, 'b l -> b l 1 1')

        # out1 = F.gelu(self.linear1(tt))
        # tt: (B, L, K, 1)
        tt = self.encoder(tt)
        tt = torch.mul(tt, self.k_map)
        return tt * non_pad_mask  # [B,L,K,D]


class Note_Tau_Encoder(nn.Module):
    def __init__(self, embed_time, hid_dim=16):
        super(Note_Tau_Encoder, self).__init__()
        self.encoder = FFNN(1, hid_dim, embed_time)

    def forward(self, tt):
        if tt.dim() == 3:  # [B,N,1]
            pass
        else:  # [B,L]
            tt = rearrange(tt, 'b n -> b n 1')

        tt = self.encoder(tt)
        return tt # [B,N,D]


class Note_Time_Encoder(nn.Module):
    def __init__(self, embed_time):
        super(Note_Time_Encoder, self).__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def forward(self, tt):
        if tt.dim() == 3:  # [B,L,1]
            pass
        else:  # [B,L]
            tt = rearrange(tt, 'b l -> b l 1')

        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        out = torch.cat([out1, out2], -1)  # [B,N,D]
        return out


class Node_Encoder(nn.Module):
    def __init__(self, d_model, out_d_model):
        super(Node_Encoder, self).__init__()
        input_dim = 5 * d_model
        output_dim = out_d_model

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.Linear(input_dim, output_dim, bias=False),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            # nn.Linear(output_dim, output_dim, bias=False),
            nn.LayerNorm(output_dim),
        )

    def forward(self, var_emb, density_emb, type_emb, time_enc_k, invar_emb, non_pad_mask):
        B, K, L, D = var_emb.shape

        type_emb = type_emb.expand(B, K, L, D)
        invar_emb = invar_emb.unsqueeze(1).unsqueeze(1)
        invar_emb = invar_emb.expand(B, K, L, D)  # * mask

        input = torch.cat([var_emb, density_emb, type_emb, time_enc_k, invar_emb], dim=-1)

        non_pad_mask = rearrange(non_pad_mask, 'b l k -> b k l 1')
        x = self.encoder(input)
        return x * non_pad_mask