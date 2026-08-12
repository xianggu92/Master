import math
import torch
from torch import nn
import torch.nn.functional as F
from core.module import TransformerCrossEncoder, gateMLP, multiTimeAttention
from core.mFand import MultiFeatureAttention


class MULTCrossModel(nn.Module):

    def __init__(self, args, device):
        super(MULTCrossModel, self).__init__()
        self.modeltype = args.modeltype
        self.num_heads = args.num_heads
        self.args = args
        self.layers = args.layers
        self.device = device
        self.kernel_size = args.kernel_size
        self.dropout = args.dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts = args.irregular_learn_emb_ts
        self.irregular_learn_emb_text = args.irregular_learn_emb_text
        self.irregular_learn_emb_cxr = args.irregular_learn_emb_cxr
        self.irregular_learn_emb_ecg = args.irregular_learn_emb_ecg
        self.use_shared_time_embed = args.use_shared_time_embed
        self.reg_ts = args.reg_ts
        self.TS_mixup = args.TS_mixup
        self.use_mFAND = args.use_mFAND
        self.mixup_level = args.mixup_level
        self.task = args.task
        self.tt_max = args.tt_max
        self.cross_method = args.cross_method
        self.num_modalities = args.num_modalities
        self.embed_dim = args.embed_dim
        self.token_type_embeddings = nn.Embedding(args.num_modalities, args.embed_dim)

        self.register_buffer('time_query', torch.linspace(0, 1.0, self.tt_max))

        if self.use_shared_time_embed:
            self.periodic = nn.Linear(1, args.embed_time - 1)
            self.linear = nn.Linear(1, 1)

        if "TS" in self.modeltype:
            self.ts_dim = args.ts_dim

            if self.irregular_learn_emb_ts == "mTAND":
                self.time_attn_ts = multiTimeAttention(self.ts_dim * 2, self.embed_dim, args.embed_time, 8, args.use_shared_time_embed)

            if self.use_mFAND:
                self.feature_attn_ts = MultiFeatureAttention(
                    embed_value=args.embed_time,
                    num_heads=self.num_heads,
                    input_value_dim=self.ts_dim * 2,
                    input_dim=self.ts_dim * 2,
                    nhidden=self.embed_dim,
                    dropout=self.dropout,
                )
                self.mfand_mtand_gate = gateMLP(
                    input_dim=self.embed_dim * 2,
                    hidden_size=self.embed_dim,
                    output_dim=self.embed_dim,
                    dropout=self.dropout,
                )

            if self.reg_ts:
                self.reg_ts_dim = args.ts_dim * 2
                self.proj_ts = nn.Conv1d(self.reg_ts_dim, self.embed_dim, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size - 1) / 2), bias=False)

            if self.TS_mixup:
                if self.mixup_level == "batch":
                    self.moe = gateMLP(input_dim=self.embed_dim * 2, hidden_size=args.embed_dim, output_dim=1, dropout=args.dropout)
                elif self.mixup_level == "batch_seq":
                    self.moe = gateMLP(input_dim=self.embed_dim * 2, hidden_size=args.embed_dim, output_dim=1, dropout=args.dropout)
                elif self.mixup_level == "batch_seq_feature":
                    self.moe = gateMLP(input_dim=self.embed_dim * 2, hidden_size=args.embed_dim, output_dim=self.embed_dim, dropout=args.dropout)
                else:
                    raise ValueError("Unknown mixedup type")

        if "Text" in self.modeltype:
            self.txt_dim = args.txt_dim

            if self.irregular_learn_emb_text == "mTAND":
                self.time_attn_text = multiTimeAttention(768, self.embed_dim, args.embed_time, 8, args.use_shared_time_embed)
            else:
                self.proj_txt = nn.Conv1d(self.txt_dim, self.embed_dim, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size - 1) / 2), bias=False)

        if "CXR" in self.modeltype:
            self.cxr_dim = args.cxr_dim

            if self.irregular_learn_emb_cxr == "mTAND":
                self.time_attn_cxr = multiTimeAttention(1024, self.embed_dim, args.embed_time, 8, args.use_shared_time_embed)
            else:
                self.proj_cxr = nn.Conv1d(self.cxr_dim, self.embed_dim, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size - 1) / 2), bias=False)

        if "ECG" in self.modeltype:
            self.ecg_dim = args.ecg_dim

            if self.irregular_learn_emb_ecg == "mTAND":
                self.time_attn_ecg = multiTimeAttention(256, self.embed_dim, args.embed_time, 8, args.use_shared_time_embed)
            else:
                self.proj_ecg = nn.Conv1d(self.ecg_dim, self.embed_dim, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size - 1) / 2), bias=False)

        dim = self.embed_dim * self.num_modalities

        if self.cross_method in ["moe", "dense"]:
            self.trans_self_cross_ts_txt = self.get_cross_network(args, layers=args.layers)
            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            self.out_layer = nn.Linear(dim, args.num_labels)

        if "ihm" in self.task or "los" in self.task:
            self.loss_fct1 = nn.CrossEntropyLoss()
        elif "pheno" in self.task:
            self.loss_fct1 = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_cross_network(self, args, layers=-1):
        embed_dim, q_seq_len = self.embed_dim, self.tt_max
        return TransformerCrossEncoder(
            args=args,
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=layers,
            device=self.device,
            attn_dropout=self.dropout,
            relu_dropout=self.dropout,
            res_dropout=self.dropout,
            embed_dropout=self.dropout,
            attn_mask=self.attn_mask,
            q_seq_len_1=q_seq_len,
            num_modalities=self.num_modalities,
        )

    def learn_time_embedding(self, tt):
        """Time2Vec Module"""
        tt = tt.to(self.device)
        tt = tt.unsqueeze(-1)
        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        return torch.cat([out1, out2], -1)

    def _missing_indices(self, missing_idx):
        all_indices = torch.arange(len(missing_idx))
        missing_indices = torch.nonzero(missing_idx).squeeze(1)
        missing_mask = torch.ones(len(missing_idx), dtype=torch.bool)
        missing_mask[missing_indices] = False
        non_missing = all_indices[missing_mask]
        return missing_indices, non_missing

    def forward(
        self,
        ts_feats,
        ts_masks,
        ts_times,
        cxr_missing=None,
        text_missing=None,
        ecg_missing=None,
        text_feats=None,
        text_times=None,
        text_masks=None,
        labels=None,
        reg_ts_feats=None,
        cxr_feats=None,
        cxr_times=None,
        cxr_masks=None,
        ecg_feats=None,
        ecg_times=None,
        ecg_masks=None,
        query_ts_feats=None,
        query_ts_masks=None,
        imputed_ts_feats=None,
        imputed_ts_masks=None,
    ):
        """dimension [batch_size, seq_len, n_features]"""
        if "TS" in self.modeltype:
            if self.irregular_learn_emb_ts == "mTAND":
                if self.use_shared_time_embed:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0))
                    time_key_ts = self.learn_time_embedding(ts_times)
                else:
                    time_query = self.time_query
                    time_key_ts = ts_times

                x_ts_irg = torch.cat((ts_feats, ts_masks), 2)
                ts_masks_expanded = torch.cat((ts_masks, ts_masks), 2)

                proj_x_ts_irg = self.time_attn_ts(time_query, time_key_ts, x_ts_irg, ts_masks_expanded)
                proj_x_ts_irg = proj_x_ts_irg.transpose(0, 1)

                if self.use_mFAND:
                    mfand_query = torch.cat((query_ts_feats, query_ts_masks), dim=-1)
                    mfand_key = mfand_value = torch.cat((imputed_ts_feats, imputed_ts_masks), dim=-1)

                    ts_emb_mask = torch.sum(ts_masks, dim=-1)
                    ts_emb_mask[ts_emb_mask > 1] = 1

                    proj_x_ts_mfand = self.feature_attn_ts(mfand_query, mfand_key, mfand_value, emb_mask=ts_emb_mask, impute_data=query_ts_feats)
                    proj_x_ts_mfand = proj_x_ts_mfand.transpose(0, 1)

                    fusion_gate = self.mfand_mtand_gate(torch.cat((proj_x_ts_irg, proj_x_ts_mfand), dim=-1))
                    proj_x_ts_irg = fusion_gate * proj_x_ts_mfand + (1 - fusion_gate) * proj_x_ts_irg

            if self.reg_ts and reg_ts_feats is not None:
                x_ts_reg = reg_ts_feats.transpose(1, 2)

                proj_x_ts_reg = x_ts_reg if self.reg_ts_dim == self.embed_dim else self.proj_ts(x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)

            if self.TS_mixup:
                if self.mixup_level == "batch":
                    g_irg = torch.max(proj_x_ts_irg, dim=0).values
                    g_reg = torch.max(proj_x_ts_reg, dim=0).values
                    moe_gate = torch.cat([g_irg, g_reg], dim=-1)
                elif self.mixup_level == "batch_seq" or self.mixup_level == "batch_seq_feature":
                    moe_gate = torch.cat([proj_x_ts_irg, proj_x_ts_reg], dim=-1)
                else:
                    raise ValueError("Unknown mixedup type")

                mixup_rate = self.moe(moe_gate)
                proj_x_ts = mixup_rate * proj_x_ts_irg + (1 - mixup_rate) * proj_x_ts_reg
            else:
                if self.irregular_learn_emb_ts:
                    proj_x_ts = proj_x_ts_irg
                elif self.reg_ts:
                    proj_x_ts = proj_x_ts_reg
                else:
                    raise ValueError("Unknown time series type")
            proj_x_ts += self.token_type_embeddings(torch.zeros((self.tt_max, ts_feats.shape[0]), dtype=torch.long, device=ts_feats.device))

        mod_count = 1
        if "Text" in self.modeltype:
            x_txt = text_feats

            if self.irregular_learn_emb_text == "mTAND":
                if self.use_shared_time_embed:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0))
                    time_key = self.learn_time_embedding(text_times)
                else:
                    time_query = self.time_query
                    time_key = text_times

                proj_x_txt = self.time_attn_text(time_query, time_key, x_txt, text_masks)
                proj_x_txt = proj_x_txt.transpose(0, 1)
            else:
                x_txt = x_txt.transpose(1, 2)
                proj_x_txt = x_txt if self.txt_dim == self.embed_dim else self.proj_txt(x_txt)
                proj_x_txt = proj_x_txt.permute(2, 0, 1)

            if text_missing is None or torch.all(text_missing == 0):
                proj_x_txt += self.token_type_embeddings(torch.ones((self.args.tt_max, ts_feats.shape[0]), dtype=torch.long, device=ts_feats.device))
            elif not torch.all(text_missing == 0):
                missing_indices, non_missing = self._missing_indices(text_missing)
                proj_x_txt[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=ts_feats.device))
                proj_x_txt[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=proj_x_txt.dtype, device=ts_feats.device)
            mod_count += 1

        if "CXR" in self.modeltype:
            if self.irregular_learn_emb_cxr == "mTAND":
                if self.use_shared_time_embed:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                    time_key = self.learn_time_embedding(cxr_times).to(self.device)
                else:
                    time_query = self.time_query
                    time_key = cxr_times

                proj_x_cxr = self.time_attn_cxr(time_query, time_key, cxr_feats, cxr_masks)
                proj_x_cxr = proj_x_cxr.transpose(0, 1)
            else:
                cxr_feats = cxr_feats.transpose(1, 2)
                proj_x_cxr = cxr_feats if self.cxr_dim == self.embed_dim else self.proj_cxr(cxr_feats)
                proj_x_cxr = proj_x_cxr.permute(2, 0, 1)

            if cxr_missing is None or torch.all(cxr_missing == 0):
                proj_x_cxr += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, ts_feats.shape[0]), dtype=torch.long, device=ts_feats.device))
            elif not torch.all(cxr_missing == 0):
                missing_indices, non_missing = self._missing_indices(cxr_missing)
                proj_x_cxr[:, non_missing, :] += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=ts_feats.device))
                proj_x_cxr[:, missing_indices, :] = torch.zeros((self.tt_max, len(missing_indices), self.args.embed_dim), dtype=proj_x_cxr.dtype, device=ts_feats.device)
            mod_count += 1

        if "ECG" in self.modeltype:
            if self.irregular_learn_emb_ecg == "mTAND":
                if self.use_shared_time_embed:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)
                    time_key = self.learn_time_embedding(ecg_times).to(self.device)
                else:
                    time_query = self.time_query
                    time_key = ecg_times

                proj_x_ecg = self.time_attn_ecg(time_query, time_key, ecg_feats, ecg_masks)
                proj_x_ecg = proj_x_ecg.transpose(0, 1)
            else:
                ecg_feats = ecg_feats.transpose(1, 2)
                proj_x_ecg = ecg_feats if self.ecg_dim == self.embed_dim else self.proj_ecg(ecg_feats)
                proj_x_ecg = proj_x_ecg.permute(2, 0, 1)

            if ecg_missing is None or torch.all(ecg_missing == 0):
                proj_x_ecg += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, ts_feats.shape[0]), dtype=torch.long, device=ts_feats.device))
            elif not torch.all(ecg_missing == 0):
                missing_indices, non_missing = self._missing_indices(ecg_missing)
                proj_x_ecg[:, non_missing, :] += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=ts_feats.device))
                proj_x_ecg[:, missing_indices, :] = torch.zeros((self.tt_max, len(missing_indices), self.args.embed_dim), dtype=proj_x_ecg.dtype, device=ts_feats.device)
            mod_count += 1

        balance_loss = None
        if self.cross_method in ["self_cross", "moe", "hme", "dense"]:
            if self.modeltype == "TS_Text":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_txt, proj_x_ts], ["txt", "ts"])
            elif self.modeltype == "TS_CXR":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_cxr, proj_x_ts], ["cxr", "ts"])
            elif self.modeltype == "TS_CXR_Text":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_cxr, proj_x_txt], ["ts", "cxr", "txt"])
            elif self.modeltype == "TS_CXR_Text_ECG":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_cxr, proj_x_txt, proj_x_ecg], ["ts", "cxr", "txt", "ecg"])
            elif self.modeltype == "TS":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_ts], ["ts"])

            if hiddens is None:
                return None
            last_hs = torch.cat([hid[-1] for hid in hiddens], dim=1)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        if "ihm" in self.task or "los" in self.task:
            if labels != None:
                task_loss = self.loss_fct1(output, labels)
                return task_loss, balance_loss
            return torch.nn.functional.softmax(output, dim=-1)[:, 1]

        elif "pheno" in self.task:
            if labels != None:
                labels = labels.float()
                task_loss = self.loss_fct1(output, labels)
                return task_loss, balance_loss
            return torch.nn.functional.sigmoid(output)
