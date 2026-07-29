import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class ScaledDotProductAttention_bias(nn.Module):

    def __init__(self, d_model, n_head, d_k, d_v, temperature, attn_dropout=0.2):
        super().__init__()

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.n_head = n_head

    def forward(self, q, k, v, mask):
        # [B,K,H,LQ,LK] for temporal, [B,L,H,Kq,Kk] for category
        # [B,K,L,H,D]
        q = rearrange(self.w_qs(q), 'b k l (n d) -> b k n l d', n=self.n_head)
        k = rearrange(self.w_ks(k), 'b k l (n d) -> b k n d l', n=self.n_head)
        v = rearrange(self.w_vs(v), 'b k l (n d) -> b k n l d', n=self.n_head)

        attn = torch.matmul(q, k) / self.temperature

        if mask is not None:
            if attn.dim() > mask.dim():
                mask = mask.unsqueeze(2).expand(attn.shape)
            attn = attn.masked_fill(mask, -1e4)

        attn = self.dropout(F.softmax(attn, dim=-1))

        v = torch.matmul(attn, v)

        v = rearrange(v, 'b k n l d -> b k l (n d)')

        return v, attn


class MultiHeadAttention_tem_bias(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc = nn.Linear(d_v * n_head, d_model)
        self.layernorm = nn.LayerNorm(d_model)  # TODO: New

        self.attention = ScaledDotProductAttention_bias(d_model, n_head, d_k, d_v, temperature=d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # event_matrix [B,L,K]

        # [B,K,H,Lq,Lk]
        output, attn = self.attention(q, k, v, mask=mask) # [B,K,H,L,D]

        output = self.dropout(self.layernorm(self.fc(output)))

        return output, attn


class MultiHeadAttention_type_bias(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc = nn.Linear(d_v * n_head, d_model)
        self.layernorm = nn.LayerNorm(d_model)  # TODO: New
        self.attention = ScaledDotProductAttention_bias(d_model, n_head, d_k, d_v, temperature=d_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # [B,L,K,D]
        output, attn = self.attention(q, k, v, mask=mask)

        output = self.dropout(self.layernorm(self.fc(output)))

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x



PAD = 0

def get_attn_key_pad_mask_K(mask, transpose=False, full_attn=False):
    """ For masking out the padding part of key sequence. """
    if full_attn:
        if transpose:
            mask = rearrange(mask, 'b l k -> b k l')
        padding_mask = repeat(mask, 'b k l1 -> b k l2 l1', l2=mask.shape[-1]).eq(PAD)
    else:
        if transpose:
            seq_q = rearrange(mask, 'b l k -> b k l 1')
            seq_k = rearrange(mask, 'b l k -> b k 1 l')
        else:
            seq_q = rearrange(mask, 'b k l -> b k l 1')
            seq_k = rearrange(mask, 'b k l -> b k 1 l')
        padding_mask = torch.matmul(seq_q, seq_k).eq(PAD)

    return padding_mask


class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, args, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.full_attn = False

        self.slf_tem_attn = MultiHeadAttention_tem_bias(
            n_head, d_model, d_k, d_v, dropout=dropout)

        self.slf_type_attn = MultiHeadAttention_type_bias(
            n_head, d_model, d_k, d_v, dropout=dropout)


        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input, non_pad_mask=None):
        # time attention
        # [B, K, L, D]
        tem_mask = get_attn_key_pad_mask_K(mask=non_pad_mask, transpose=False, full_attn=self.full_attn)
        type_mask = get_attn_key_pad_mask_K(mask=non_pad_mask, transpose=True, full_attn=self.full_attn)

        # residue = enc_input
        tem_output = self.layer_norm(input)

        tem_output, enc_tem_attn = self.slf_tem_attn(tem_output, tem_output, tem_output, mask=tem_mask)

        tem_output = tem_output + input

        tem_output = rearrange(tem_output, 'b k l d -> b l k d')

        # type attention
        # [B, L, K, D]
        # residue = enc_output
        type_output = self.layer_norm(tem_output)

        type_output, enc_type_attn = self.slf_type_attn(type_output, type_output, type_output, mask=type_mask)

        enc_output = type_output + tem_output

        # FFFNN
        # residue = enc_output
        output = self.layer_norm(enc_output)

        output = self.pos_ffn(output)

        output = output + enc_output

        output = rearrange(output, 'b l k d -> b k l d')

        # optional
        output = self.layer_norm(output)

        return output, enc_tem_attn, enc_type_attn


class Attention(nn.Module):

    def __init__(self, hin_d, d_model):
        super().__init__()

        self.linear = nn.Linear(d_model, hin_d)
        self.W = nn.Linear(hin_d, 1, bias=False)

    def forward(self, x, mask=None, mask_value=-1e30):
        # [B,L,K,D]

        # map directly
        attn = self.W(torch.tanh(self.linear(x)))  # [B,L,K,1]

        if mask is not None:
            attn = mask * attn + (1 - mask) * mask_value
            # attn = attn.masked_fill(mask, mask_value)

        attn = F.softmax(attn, dim=-2)

        x = torch.matmul(x.transpose(-1, -2), attn).squeeze(-1)  # [B,L,D,1]

        return x, attn

class Attention_Aggregator(nn.Module):
    '''[B K L D--> B L D]'''
    def __init__(self, dim, out_dim):
        super(Attention_Aggregator, self).__init__()

        # self.linear = nn.Linear(dim, out_dim)
        self.attention_type = Attention(out_dim*2, out_dim)

    def forward(self, ENCoutput, mask):
        """
        input: [B,K,L,D], mask: [B,K,L,1]
        """
        mask = rearrange(mask, 'b k l 1 -> b l k 1')
        ts_mask = torch.sum(mask, dim=2)  # [B L 1]
        ts_mask[ts_mask > 1] = 1
        ENCoutput = rearrange(ENCoutput, 'b k l d -> b l k d')
        # ENCoutput = self.linear(ENCoutput)
        ENCoutput, _ = self.attention_type(ENCoutput, mask) # [B L D]
        ENCoutput = ENCoutput * ts_mask
        # [B K L D] --> [B D]
        # ENCoutput, _ = self.attention_len(ENCoutput, mask) # [B,K,D]
        # ENCoutput, _ = self.attention_type(ENCoutput) # [B,D]
        return ENCoutput