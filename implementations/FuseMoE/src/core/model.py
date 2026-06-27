import torch
from torch import nn
import torch.nn.functional as F
import math
from core.module import *
from core.interp import *
from core.patch_interpolation import PatchInterpolation


class BertForRepresentation(nn.Module):
    """
    This class represents a BERT model for text representation.

    Args:
        args (object): The arguments for the model.
        BioBert (object): The BioBERT model.

    Attributes:
        bert (object): The BioBERT model.
        dropout (object): The dropout layer.
        model_name (str): The name of the model.
    """
    
    def __init__(self, args,BioBert):
        super().__init__()
        self.bert = BioBert

        self.dropout = torch.nn.Dropout(BioBert.config.hidden_dropout_prob)
        self.model_name=args.model_name

    def forward(self, input_ids_sequence, attention_mask_sequence, sent_idx_list=None , doc_idx_list=None):
        """
        Forward pass of the model.

        Args:
            input_ids_sequence (List[Tensor]): List of input token IDs for each sequence.
            attention_mask_sequence (List[Tensor]): List of attention masks for each sequence.
            sent_idx_list (List[int], optional): List of sentence indices. Defaults to None.
            doc_idx_list (List[int], optional): List of document indices. Defaults to None.

        Returns:
            Tensor: Text embeddings for each sequence.
        """
        txt_arr = []

        for input_ids, attention_mask in zip(input_ids_sequence, attention_mask_sequence):

            if 'Longformer' in self.model_name:

                attention_mask-=1

                text_embeddings=self.bert(input_ids, global_attention_mask=attention_mask)
            else:
                text_embeddings=self.bert(input_ids, attention_mask=attention_mask)
            text_embeddings= text_embeddings[0][:,0,:]
            text_embeddings = self.dropout(text_embeddings)
            txt_arr.append(text_embeddings)

        txt_arr=torch.stack(txt_arr)
        return txt_arr


class MULTCrossModel(nn.Module):
    def __init__(self, args, device, modeltype=None, orig_d_ts=None, orig_reg_d_ts=None, orig_d_txt=None, ts_seq_num=None):
        """
        Construct a MulT Cross model.
        """
        super(MULTCrossModel, self).__init__()
        if modeltype!=None:
            self.modeltype = modeltype
        else:
            self.modeltype = args.modeltype
        self.num_heads = args.num_heads
        self.args = args
        self.layers = args.layers
        self.device = device
        self.kernel_size=args.kernel_size
        self.dropout = args.dropout
        self.attn_mask = False
        self.irregular_learn_emb_ts = args.irregular_learn_emb_ts
        self.irregular_learn_emb_text = args.irregular_learn_emb_text
        self.irregular_learn_emb_cxr = args.irregular_learn_emb_cxr
        self.irregular_learn_emb_ecg = args.irregular_learn_emb_ecg
        self.reg_ts = args.reg_ts
        self.TS_mixup = args.TS_mixup
        self.mixup_level = args.mixup_level
        self.task = args.task
        self.tt_max = args.tt_max
        self.n_ref_point = args.n_ref_point
        self.cross_method = args.cross_method
        self.num_modalities = args.num_modalities
        self.token_type_embeddings = nn.Embedding(args.num_modalities, args.embed_dim)

        if self.irregular_learn_emb_ts is not None or self.irregular_learn_emb_text is not None:
            self.time_query = torch.linspace(0, 1., self.n_ref_point)
            self.periodic = nn.Linear(1, args.embed_time - 1)
            self.linear = nn.Linear(1, 1)

        if "TS" in self.modeltype:
            self.orig_d_ts=orig_d_ts
            self.d_ts=args.embed_dim
            self.ts_seq_num=ts_seq_num

            if self.irregular_learn_emb_ts == 'mTAND':
                self.time_attn_ts = multiTimeAttention(self.orig_d_ts*2, self.d_ts, args.embed_time, 8)
            elif self.irregular_learn_emb_ts == 'PatchInterpolation':
                self.patch_interpolation_ts = PatchInterpolation(self.orig_d_ts*2, self.d_ts, args.embed_time, 8, args.tt_max, args.n_patch, args.n_ref_point)
 
            if self.reg_ts:
                self.orig_reg_d_ts=orig_reg_d_ts
                self.proj_ts = nn.Conv1d(self.orig_reg_d_ts, self.d_ts, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

            if self.TS_mixup:
                if self.mixup_level=='batch':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
                elif self.mixup_level=='batch_seq':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=1,dropout=args.dropout)
                elif self.mixup_level=='batch_seq_feature':
                    self.moe =gateMLP(input_dim=self.d_ts*2,hidden_size=args.embed_dim,output_dim=self.d_ts,dropout=args.dropout)
                else:
                    raise ValueError("Unknown mixedup type")

        if "Text" in self.modeltype:
            self.orig_d_txt = orig_d_txt
            self.d_txt = args.embed_dim

            if self.irregular_learn_emb_text == 'mTAND':
                self.time_attn_text = multiTimeAttention(768, self.d_txt, args.embed_time, 8)
            elif self.irregular_learn_emb_ts == 'PatchInterpolation':
                self.patch_interpolation_txt = PatchInterpolation(768, self.d_txt, args.embed_time, 8, args.tt_max, args.n_patch, args.n_ref_point)
            else:
                self.proj_txt = nn.Conv1d(self.orig_d_txt, self.d_txt, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if "CXR" in self.modeltype:
            self.orig_d_cxr = 1024
            self.d_cxr = args.embed_dim

            if self.irregular_learn_emb_cxr == 'mTAND':
                self.time_attn_cxr = multiTimeAttention(1024, self.d_cxr, args.embed_time, 8)
            else:
                self.proj_cxr = nn.Conv1d(self.orig_d_cxr, self.d_cxr, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        if "ECG" in self.modeltype:
            self.orig_d_ecg = 256
            self.d_ecg = args.embed_dim

            if self.irregular_learn_emb_ecg == 'mTAND':
                self.time_attn_ecg = multiTimeAttention(256, self.d_ecg, args.embed_time, 8)
            else:
                self.proj_ecg = nn.Conv1d(self.orig_d_ecg, self.d_ecg, kernel_size=self.kernel_size, padding=math.floor((self.kernel_size -1) / 2), bias=False)

        output_dim = args.num_labels

        if self.cross_method in ["self_cross", "moe", "hme"]:
            self.trans_self_cross_ts_txt = self.get_cross_network(args, layers=args.layers)
            dim = 0
            if "TS" in self.modeltype:
                dim += self.d_ts
            if "Text" in self.modeltype:
                dim += self.d_txt
            if "CXR" in self.modeltype:
                dim += self.d_cxr
            if "ECG" in self.modeltype:
                dim += self.d_ecg            

            self.proj1 = nn.Linear(dim, dim)
            self.proj2 = nn.Linear(dim, dim)
            self.out_layer = nn.Linear(dim, output_dim)

        if 'ihm' in self.task or 'los' in self.task:
            self.loss_fct1 = nn.CrossEntropyLoss()
        elif 'pheno' in self.task:
            self.loss_fct1 = nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown task")

    def get_cross_network(self, args, layers=-1):
        embed_dim, q_seq_len = self.d_ts, self.tt_max
        return TransformerCrossEncoder(args=args,
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
                                        num_modalities=self.num_modalities)

    def learn_time_embedding(self, tt):
        '''
        Time2Vec Module
        '''
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

    def forward(self, x_ts, x_ts_mask, ts_tt_list, cxr_missing=None, text_missing=None, ecg_missing=None, input_ids_sequences=None,
                attn_mask_sequences=None, text_emb=None, note_time_list=None, note_time_mask_list=None,
                labels=None, reg_ts=None, cxr_feats=None, cxr_time=None, cxr_time_mask=None, ecg_feats=None,
                ecg_time=None, ecg_time_mask=None, **kwargs):
        """
        dimension [batch_size, seq_len, n_features]

        """
        if "TS" in self.modeltype:
            if self.irregular_learn_emb_ts == 'mTAND':
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0))
                time_key_ts = self.learn_time_embedding(ts_tt_list)

                x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)

                proj_x_ts_irg = self.time_attn_ts(time_query, time_key_ts, x_ts_irg, x_ts_mask)
                proj_x_ts_irg = proj_x_ts_irg.transpose(0, 1)

            elif self.irregular_learn_emb_ts == 'PatchInterpolation':
                # 需要展開 Query 的 Batch 維度，之後分塊才能跟 Key 的 Batch 維度對齊
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).expand(ts_tt_list.shape[0], -1, -1)
                time_key_ts = self.learn_time_embedding(ts_tt_list)

                x_ts_irg = torch.cat((x_ts, x_ts_mask), 2)
                x_ts_mask = torch.cat((x_ts_mask, x_ts_mask), 2)

                proj_x_ts_irg = self.patch_interpolation_ts(time_query, time_key_ts, x_ts_irg, ts_tt_list, x_ts_mask)
                proj_x_ts_irg = proj_x_ts_irg.transpose(0, 1)

            if self.reg_ts and reg_ts != None: 
                x_ts_reg = reg_ts.transpose(1, 2)

                proj_x_ts_reg = x_ts_reg if self.orig_reg_d_ts == self.d_ts else self.proj_ts(x_ts_reg)
                proj_x_ts_reg = proj_x_ts_reg.permute(2, 0, 1)


            if self.TS_mixup:
                if self.mixup_level=='batch':
                    g_irg=torch.max(proj_x_ts_irg, dim=0).values
                    g_reg =torch.max(proj_x_ts_reg, dim=0).values
                    moe_gate=torch.cat([g_irg, g_reg], dim=-1)
                elif self.mixup_level=='batch_seq' or  self.mixup_level=='batch_seq_feature':
                    moe_gate=torch.cat([proj_x_ts_irg,proj_x_ts_reg],dim=-1)
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
            proj_x_ts += self.token_type_embeddings(torch.zeros((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))

        mod_count = 1
        if "Text" in self.modeltype:
            # compute irregular clinical notes attention
            x_txt = text_emb

            if self.irregular_learn_emb_text == 'mTAND':
                time_key = self.learn_time_embedding(note_time_list)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                proj_x_txt = self.time_attn_text(time_query, time_key, x_txt, note_time_mask_list)
                proj_x_txt = proj_x_txt.transpose(0, 1)
            elif self.irregular_learn_emb_text == 'PatchInterpolation':
                time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).expand(ts_tt_list.shape[0], -1, -1)
                time_key = self.learn_time_embedding(note_time_list)

                proj_x_txt = self.patch_interpolation_txt(time_query, time_key, x_txt, note_time_list, note_time_mask_list)
                proj_x_txt = proj_x_txt.transpose(0, 1)
            else:
                x_txt = x_txt.transpose(1, 2)
                proj_x_txt = x_txt if self.orig_d_txt == self.d_txt else self.proj_txt(x_txt)
                proj_x_txt = proj_x_txt.permute(2, 0, 1)

            if text_missing is None or torch.all(text_missing == 0):
                proj_x_txt += self.token_type_embeddings(torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
            elif not torch.all(text_missing == 0):
                missing_indices, non_missing = self._missing_indices(text_missing)
                proj_x_txt[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_txt[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
            mod_count += 1

        if "CXR" in self.modeltype:
            # compute irregular clinical notes attention
            if self.irregular_learn_emb_cxr:
                time_key = self.learn_time_embedding(cxr_time).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                proj_x_cxr=self.time_attn_cxr(time_query, time_key, cxr_feats, cxr_time_mask)
                proj_x_cxr=proj_x_cxr.transpose(0, 1)
            else:
                cxr_feats = cxr_feats.transpose(1, 2)
                proj_x_cxr = cxr_feats if self.orig_d_cxr == self.d_cxr else self.proj_cxr(cxr_feats)
                proj_x_cxr = proj_x_cxr.permute(2, 0, 1)
            if cxr_missing is None or torch.all(cxr_missing == 0):
                proj_x_cxr += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
            elif not torch.all(cxr_missing == 0):
                # proj_x_cxr = None
                missing_indices, non_missing = self._missing_indices(cxr_missing)
                proj_x_cxr[:, non_missing, :] += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_cxr[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
            mod_count += 1

        if "ECG" in self.modeltype:
            # compute irregular ECG attention
            if self.irregular_learn_emb_ecg:
                time_key = self.learn_time_embedding(ecg_time).to(self.device)
                if not self.irregular_learn_emb_ts:
                    time_query = self.learn_time_embedding(self.time_query.unsqueeze(0)).to(self.device)

                proj_x_ecg=self.time_attn_ecg(time_query, time_key, ecg_feats, ecg_time_mask)
                proj_x_ecg=proj_x_ecg.transpose(0, 1)
            else:
                ecg_feats = ecg_feats.transpose(1, 2)
                proj_x_ecg = ecg_feats if self.orig_d_ecg == self.d_ecg else self.proj_ecg(ecg_feats)
                proj_x_ecg = proj_x_ecg.permute(2, 0, 1)
            
            if ecg_missing is None or torch.all(ecg_missing == 0):
                proj_x_ecg += self.token_type_embeddings(mod_count * torch.ones((self.args.tt_max, x_ts.shape[0]), dtype=torch.long, device=x_ts.device))
            elif not torch.all(ecg_missing == 0):
                # proj_x_ecg = None
                missing_indices, non_missing = self._missing_indices(ecg_missing)
                proj_x_ecg[:, non_missing, :] += self.token_type_embeddings(torch.ones((self.args.tt_max, len(non_missing)), dtype=torch.long, device=x_ts.device))
                proj_x_ecg[:, missing_indices, :] = torch.zeros((self.args.tt_max, len(missing_indices), self.args.embed_dim), dtype=torch.float16, device=x_ts.device)
            mod_count += 1

        balance_loss = None
        if self.cross_method in ["self_cross", "moe", "hme"]:
            if self.modeltype == "TS_Text":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_txt, proj_x_ts], ['txt', 'ts'])
            elif self.modeltype == "TS_CXR":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_cxr, proj_x_ts], ['cxr', 'ts'])
            elif self.modeltype == "TS_CXR_Text":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_cxr, proj_x_txt], ['ts', 'cxr', 'txt'])
            elif self.modeltype == "TS_CXR_Text_ECG":
                hiddens, balance_loss = self.trans_self_cross_ts_txt([proj_x_ts, proj_x_cxr, proj_x_txt, proj_x_ecg], ['ts', 'cxr', 'txt', 'ecg'])

            if hiddens is None:
                return None
            # h_txt_with_ts, h_ts_with_txt=hiddens
            last_hs = torch.cat([hid[-1] for hid in hiddens], dim=1)
            # last_hs = torch.cat([h_txt_with_ts[-1], h_ts_with_txt[-1]], dim=1)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.dropout, training=self.training))
        last_hs_proj += last_hs
        output = self.out_layer(last_hs_proj)

        if 'ihm' in self.task or 'los' in self.task:
            if labels!=None:
                task_loss = self.loss_fct1(output, labels)
                return task_loss, balance_loss
            return torch.nn.functional.softmax(output,dim=-1)[:,1]

        elif 'pheno' in self.task:
            if labels!=None:
                labels=labels.float()
                task_loss = self.loss_fct1(output, labels)
                return task_loss, balance_loss
            return torch.nn.functional.sigmoid(output)