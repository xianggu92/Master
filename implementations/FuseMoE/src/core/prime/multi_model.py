import torch
from torch import nn
from core.prime.TS_dual_attention import EncoderLayer, Attention_Aggregator
from core.prime.encoder import Time_Encoder, Note_Tau_Encoder, Note_Time_Encoder, Var_Encoder, Type_Encoder, \
    Density_Encoder, Node_Encoder
from einops import rearrange, repeat


class TSModel(nn.Module):
    def __init__(self, args):
        super(TSModel, self).__init__()

        self.args = args

        self.embed_dim = args.embed_dim
        self.var_dim = args.ts_dim

        # var encoder
        self.var_enc = Var_Encoder(output_dim=self.embed_dim, num_type=self.var_dim)

        # type encoder
        self.type_enc = Type_Encoder(d_model=self.embed_dim, num_types=self.var_dim)
        self.type_matrix = torch.tensor([int(i) for i in range(self.var_dim)]).to(torch.int)
        self.type_matrix = rearrange(self.type_matrix, 'k -> 1 1 k')

        # time encoder
        self.learn_time_embedding = Time_Encoder(self.embed_dim, self.var_dim)

        # density encoder
        self.density_encoder = Density_Encoder(self.embed_dim, self.var_dim)

        # node encoder
        self.node_encoder = Node_Encoder(self.embed_dim, self.embed_dim)

        # dual attention
        self.dual_attention_stack = nn.ModuleList([
            EncoderLayer(args=args, d_model=self.embed_dim, d_inner=int(self.embed_dim/2), n_head=args.num_heads,\
                            d_k=self.embed_dim//args.num_heads, d_v=self.embed_dim//args.num_heads, dropout=args.dropout)
            for _ in range(args.ts_dual_attention_layer)
        ])

        # agg_attn
        self.agg_attention = Attention_Aggregator(self.embed_dim, self.embed_dim)


    def forward(self, ts_data, ts_tt, ts_mask, ts_tau):
        # ts data: B L K
        # ts_tt: B L
        # ts_mask: B L K
        # ts_tau: B L
        B, L, K = ts_data.shape

        # continue var emb
        var_emb = self.var_enc(ts_data, ts_mask)
        var_emb = rearrange(var_emb, 'b l k d -> b k l d')  # [B,K-cate,L,D]

        # type emb
        type_emb = self.type_matrix.to(ts_data.device)  # 1 1 K
        type_emb = self.type_enc(type_emb)  # 1 1 K D
        type_emb = rearrange(type_emb, 'b l k d -> b k l d')  # [B,K,L,D]

        # time emb
        time_enc_k = self.learn_time_embedding(ts_tt, ts_mask)  # [B,L], [B,L,K]-->[B,L,K,D]
        time_enc_k = rearrange(time_enc_k, 'b l k d -> b k l d')  # [B,K,L,D]

        # density emb
        density_emb = self.density_encoder(ts_tau, ts_mask)
        density_emb = rearrange(density_emb, 'b l k d -> b k l d')

        h0 = var_emb + density_emb + time_enc_k + type_emb
        ts_mask = rearrange(ts_mask, 'b l k -> b k l')


        ts_emb_mask = torch.sum(ts_mask, dim=1)  # B L-l
        ts_emb_mask[ts_emb_mask>1] = 1

        # dual attention + agg
        z0 = None
        for i, dual_attention in enumerate(self.dual_attention_stack):
            if i > 0 and self.args.full_attn:
                ts_mask = torch.ones_like(ts_mask).to(ts_mask.device)

            h0, _, _ = dual_attention(h0, ts_mask)

            output = self.agg_attention(h0, rearrange(ts_mask, 'b k l -> b k l 1'))  # [B K L D] --> [B L D]

            if z0 is not None and z0.shape == output.shape:
                z0 = z0 + output
            else:
                z0 = output

        return z0, ts_emb_mask  # z0 [B L-l D]/[B L-l K] ts_emb_mask [B L-l]/[B L-l 3*K]


class ImputeTSEncoder(nn.Module):
    def __init__(self,args):
        """
        Construct a Impute ts Encoder
        """
        super(ImputeTSEncoder, self).__init__()

        self.args = args

        self.ts_embed_dim = args.ts_embed_dim
        self.var_dim = args.var_dim

        # encoder
        # var encoder
        self.var_enc = nn.Sequential(
                nn.Linear(args.var_dim, args.ts_embed_dim),
                nn.LayerNorm(args.ts_embed_dim),
                nn.ReLU(),
                nn.Linear(args.ts_embed_dim, args.ts_embed_dim),
            )

    def forward(self, invar_emb, query_ts_data, query_ts_tt):
        # invar_emb: B D
        # query_ts_data: B L_t+l K
        # query_ts_tt: B L_t+l

        # var emb
        var_emb = self.var_enc(query_ts_data)  # B L_t+l D

        # time emb
        # time_enc_k = self.learn_time_embedding(query_ts_tt)  # [B L_t+l]-->[B L_t+l,D]

        # invar emb
        # invar_emb = invar_emb.unsqueeze(1)  # B D -> B 1 D

        query_ts_rep = var_emb  # B L_t+l D

        return query_ts_rep