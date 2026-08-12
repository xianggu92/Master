import os
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np


def data_prepare(args, mode, data=None):
    """Prepare the data for training or evaluation."""
    dataset = TSNoteIrgDataset(args, mode, data)

    if mode == "train":
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, collate_fn=text_ts_irg_collate_fn)
    else:
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, collate_fn=text_ts_irg_collate_fn)

    return dataset, sampler, dataloader


def impute_missing_values(features, timestamps, feature_mask, duration, max_time):
    """Imputes missing values in the input data based on the discretization rule mentioned in the paper."""
    num_features = features.shape[1]
    imputed_data = np.zeros(shape=(max_time // duration, num_features * 2))

    for feat_row, time_val, mask_row in zip(features, timestamps, feature_mask):
        target_row_idx = int(time_val / duration)
        if target_row_idx >= max_time:
            continue

        for feat_idx, (feat_val, is_present) in enumerate(zip(feat_row, mask_row)):
            if is_present == 1:
                imputed_data[target_row_idx][num_features + feat_idx] = 1
                imputed_data[target_row_idx][feat_idx] = feat_val
            else:
                if imputed_data[target_row_idx - 1][feat_idx] != 0:
                    imputed_data[target_row_idx][feat_idx] = imputed_data[target_row_idx - 1][feat_idx]

    return imputed_data


def impute_ts(query_ts_tt, ts_data, ts_mask, ts_tt, sort="+"):
    # print("tsdata:", ts_data)
    # print("ts_mask:", ts_mask)
    ts_data = ts_data * ts_mask
    L, K = ts_data.shape
    L_t = query_ts_tt.shape[0]
    query_ts_data = torch.zeros((L_t, K), dtype=ts_data.dtype).to(ts_data.device)
    query_ts_mask = torch.ones((L_t, K), dtype=ts_data.dtype).to(ts_data.device) * 0.5
    query_ts_dt = torch.zeros((L_t, K), dtype=ts_data.dtype).to(ts_data.device)
    mean_data = torch.sum(ts_data, dim=0) / torch.sum(ts_mask, dim=0)
    mean_data[mean_data.isnan()] = 0

    k_sum = torch.sum(ts_mask, dim=0)
    # print("k_sum:", k_sum)
    # query_ts_mask[:, k_sum==0] = 0
    # print("query_mask:", query_ts_mask)

    if sort == "+":
        ts_data_index = 0
        query_tt_index = 0
        # 先把范围外的用均值补上，dt为0
        while query_ts_tt[query_tt_index] < ts_tt[ts_data_index]:
            query_ts_data[query_tt_index] = mean_data
            query_ts_dt[query_tt_index] = torch.zeros((K), dtype=ts_data.dtype).to(ts_data.device)
            query_tt_index += 1
            if query_tt_index >= L_t:
                break

        # 中间的 依次替换 保留上一个有效值及上一个有效值的时刻
        now_ts_data = mean_data
        now_ts_mask = torch.ones((K), dtype=ts_data.dtype).to(ts_data.device) * 0.5
        now_ts_mask[k_sum==0] = 0
        now_ts_index_data = ts_data[ts_data_index]
        now_ts_data[ts_mask[ts_data_index]==1] = now_ts_index_data[ts_mask[ts_data_index]==1]
        now_ts_mask[ts_mask[ts_data_index]==1] = 1
        now_ts_data_tt = torch.ones((K), dtype=ts_tt.dtype).to(ts_tt.device) * ts_tt[ts_data_index]
        while ts_data_index < L-1 and query_tt_index < L_t:
            if query_ts_tt[query_tt_index] >= ts_tt[ts_data_index] and query_ts_tt[query_tt_index] < ts_tt[ts_data_index + 1]:
                query_ts_data[query_tt_index] = now_ts_data
                query_ts_mask[query_tt_index] = now_ts_mask
                query_ts_dt[query_tt_index] = query_ts_tt[query_tt_index] - now_ts_data_tt
                query_tt_index += 1
                continue
            ts_data_index += 1
            now_ts_data[ts_mask[ts_data_index]==1] = ts_data[ts_data_index, ts_mask[ts_data_index]==1]
            now_ts_mask[ts_mask[ts_data_index]==1] = 1
            now_ts_data_tt[ts_mask[ts_data_index]==1] = ts_tt[ts_data_index]
        # 若超出ts tt，则继续。
        while query_tt_index < L_t and query_ts_tt[query_tt_index] >= ts_tt[ts_data_index]:
            query_ts_data[query_tt_index] = now_ts_data
            query_ts_mask[query_tt_index] = now_ts_mask
            query_ts_dt[query_tt_index] = query_ts_tt[query_tt_index] - now_ts_data_tt
            query_tt_index += 1

    if sort == "-":
        ts_data_index = L-1
        query_tt_index = L_t-1
        while query_ts_tt[query_tt_index] > ts_tt[ts_data_index]:
            query_ts_data[query_tt_index] = mean_data
            query_ts_dt[query_tt_index] = torch.zeros((K), dtype=ts_data.dtype).to(ts_data.device)
            query_tt_index -= 1
            if query_tt_index < 0:
                break

        now_ts_data = mean_data
        now_ts_mask = torch.ones((K), dtype=ts_data.dtype).to(ts_data.device) * 0.5
        now_ts_mask[k_sum == 0] = 0
        now_ts_index_data = ts_data[ts_data_index]
        now_ts_data[ts_mask[ts_data_index] == 1] = now_ts_index_data[ts_mask[ts_data_index] == 1]
        now_ts_mask[ts_mask[ts_data_index] == 1] = 1
        now_ts_data_tt = torch.ones((K), dtype=ts_tt.dtype).to(ts_tt.device) * ts_tt[ts_data_index]
        while ts_data_index > 0 and query_tt_index > -1:
            if query_ts_tt[query_tt_index] <= ts_tt[ts_data_index] and query_ts_tt[query_tt_index] > ts_tt[
                ts_data_index - 1]:
                query_ts_data[query_tt_index] = now_ts_data
                query_ts_mask[query_tt_index] = now_ts_mask
                query_ts_dt[query_tt_index] = now_ts_data_tt - query_ts_tt[query_tt_index]
                query_tt_index -= 1
                continue
            ts_data_index -= 1
            now_ts_data[ts_mask[ts_data_index] == 1] = ts_data[ts_data_index, ts_mask[ts_data_index] == 1]
            now_ts_mask[ts_mask[ts_data_index] == 1] = 1
            now_ts_data_tt[ts_mask[ts_data_index] == 1] = ts_tt[ts_data_index]

        while query_tt_index > -1 and query_ts_tt[query_tt_index] <= ts_tt[ts_data_index]:
            query_ts_data[query_tt_index] = now_ts_data
            query_ts_mask[query_tt_index] = now_ts_mask
            query_ts_dt[query_tt_index] = now_ts_data_tt - query_ts_tt[query_tt_index]
            query_tt_index -= 1

    return query_ts_data, query_ts_dt, query_ts_mask


def impute_ts_data(query_ts_tt, ts_data, ts_mask, ts_tt):
    query_ts_data, _, query_ts_mask = impute_ts(query_ts_tt, ts_data, ts_mask, ts_tt, sort="+")
    imputed_ts_data, _, imputed_ts_mask = impute_ts(ts_tt, ts_data, ts_mask, ts_tt, sort="+")

    return query_ts_data, query_ts_mask, imputed_ts_data, imputed_ts_mask


class TSNoteIrgDataset(Dataset):
    """A PyTorch dataset class for handling time series note data in the MIMIC-IV dataset."""

    def __init__(self, args, mode, data=None):
        if data is not None:
            self.data = data
        else:
            self.data = load_data(file_path=args.file_path, mode=mode, task=args.task)
        self.args = args
        self.model_type = args.modeltype
        self.mode = mode
        self.max_time = args.tt_max
        self.reg_ts = args.reg_ts
        self.impute = args.impute
        self.use_mFAND = getattr(args, "use_mFAND", False)
        if args.debug:
            self.data = self.data[:100]

    def __getitem__(self, idx):
        sample_dict = {}
        data_detail = self.data[idx]

        label = torch.tensor(data_detail["label"], dtype=torch.long)
        sample_dict["label"] = label

        if "TS" in self.model_type:
            if self.impute:
                ts_features = data_detail['irg_ts_imputed']
                ts_mask = np.ones(ts_features.shape)
            else:
                ts_features = data_detail['irg_ts']
                ts_mask = data_detail["irg_ts_mask"]

            ts_timestamps = data_detail["ts_tt"].astype(np.float32)

            if self.reg_ts:
                regularized_ts = impute_missing_values(ts_features, ts_timestamps, ts_mask, 1, self.max_time)
                regularized_ts = torch.tensor(regularized_ts, dtype=torch.float)
            else:
                regularized_ts = None

            ts_features = torch.tensor(ts_features, dtype=torch.float)
            ts_mask = torch.tensor(ts_mask, dtype=torch.float)
            ts_timestamps = torch.tensor(ts_timestamps / self.max_time, dtype=torch.float)

            if self.use_mFAND:
                query_tt = torch.linspace(0, 1.0, self.max_time)
                query_data, query_mask, imputed_data, imputed_mask = impute_ts_data(
                    query_tt, ts_features, ts_mask, ts_timestamps
                )
                sample_dict["query_ts_feat"] = query_data
                sample_dict["query_ts_mask"] = query_mask
                sample_dict["imputed_ts_feat"] = imputed_data
                sample_dict["imputed_ts_mask"] = imputed_mask

            sample_dict["ts_feat"] = ts_features
            sample_dict["ts_time"] = ts_timestamps
            sample_dict["ts_mask"] = ts_mask
            sample_dict["reg_ts_feat"] = regularized_ts

        if "Text" in self.model_type:
            if not data_detail["text_missing"]:
                text_embeddings = data_detail["text_embeddings"]
                text_embeddings = torch.tensor(np.array(text_embeddings), dtype=torch.float)

                text_time_to_end = data_detail["text_time_to_end"].astype(np.float32)
                text_time_to_end = torch.tensor(text_time_to_end / self.max_time, dtype=torch.float)

                text_time_mask = torch.tensor([1] * len(text_time_to_end), dtype=torch.float)
            else:
                text_embeddings = torch.zeros((1, 768))
                text_time_to_end = torch.zeros(1)
                text_time_mask = torch.ones(1)

            sample_dict["text_feat"] = text_embeddings
            sample_dict["text_time"] = text_time_to_end
            sample_dict["text_mask"] = text_time_mask
            sample_dict["text_missing"] = data_detail["text_missing"]
            sample_dict["text_raw_data"] = data_detail["text_data"]

        if "CXR" in self.model_type:
            if not data_detail["cxr_missing"]:
                cxr_feats = data_detail["cxr_feats"]
                cxr_feats = torch.tensor(np.array(cxr_feats), dtype=torch.float)

                cxr_time_to_end = data_detail["cxr_time"].astype(np.float32)
                cxr_time_to_end = torch.tensor(cxr_time_to_end / self.max_time, dtype=torch.float)

                cxr_time_mask = torch.tensor([1] * len(cxr_time_to_end), dtype=torch.float)
            else:
                cxr_feats = torch.zeros((1, 1024))
                cxr_time_to_end = torch.zeros(1)
                cxr_time_mask = torch.ones(1)

            sample_dict["cxr_feat"] = cxr_feats
            sample_dict["cxr_time"] = cxr_time_to_end
            sample_dict["cxr_mask"] = cxr_time_mask
            sample_dict["cxr_missing"] = data_detail["cxr_missing"]

        if "ECG" in self.model_type:
            if not data_detail["ecg_missing"]:
                ecg_feats = data_detail["ecg_feats"]
                ecg_feats = torch.tensor(np.array(ecg_feats), dtype=torch.float).squeeze(1)

                # If any ecg_feats are nan, replace with 0
                ecg_feats[torch.isnan(ecg_feats)] = 0

                # If any ecg_feats are inf, replace with 0
                ecg_feats[torch.isinf(ecg_feats)] = 0

                ecg_time_to_end = data_detail["ecg_time"].astype(np.float32)
                ecg_time_to_end = torch.tensor(ecg_time_to_end / self.max_time, dtype=torch.float)

                ecg_time_mask = torch.tensor([1] * len(ecg_time_to_end), dtype=torch.float)
            else:
                ecg_feats = torch.zeros((1, 256))
                ecg_time_to_end = torch.zeros(1)
                ecg_time_mask = torch.ones(1)

            sample_dict["ecg_feat"] = ecg_feats
            sample_dict["ecg_time"] = ecg_time_to_end
            sample_dict["ecg_mask"] = ecg_time_mask
            sample_dict["ecg_missing"] = data_detail["ecg_missing"]

        return sample_dict

    def __len__(self):
        return len(self.data)


def load_data(file_path, mode, text=False, task="ihm"):
    """Load data from a file."""
    data_path = os.path.join(file_path, f"{mode}_{task}_stays.pkl")
    data = None
    if os.path.isfile(data_path):
        print("Using", data_path)
        with open(data_path, "rb") as f:
            data = pickle.load(f)
    return data


def text_ts_irg_collate_fn(batch):
    batch_output = {}

    if "text_missing" in batch[0].keys():
        batch_output["text_missing"] = torch.stack([torch.tensor(example["text_missing"]) for example in batch])

    if "cxr_missing" in batch[0].keys():
        batch_output["cxr_missing"] = torch.stack([torch.tensor(example["cxr_missing"]) for example in batch])

    if "ecg_missing" in batch[0].keys():
        batch_output["ecg_missing"] = torch.stack([torch.tensor(example["ecg_missing"]) for example in batch])

    try:
        ts_input_sequences = pad_sequence([example["ts_feat"] for example in batch], batch_first=True, padding_value=0)
        ts_mask_sequences = pad_sequence([example["ts_mask"] for example in batch], batch_first=True, padding_value=0)
        ts_timestamps = pad_sequence([example["ts_time"] for example in batch], batch_first=True, padding_value=0)
        labels = torch.stack([example["label"] for example in batch])

        if batch[0]["reg_ts_feat"] is not None:
            reg_ts_input = torch.stack([example["reg_ts_feat"] for example in batch])
        else:
            reg_ts_input = None

        batch_output["ts_feats"] = ts_input_sequences
        batch_output["ts_masks"] = ts_mask_sequences
        batch_output["ts_times"] = ts_timestamps
        batch_output["reg_ts_feats"] = reg_ts_input
        batch_output["labels"] = labels

        if "query_ts_feat" in batch[0]:
            batch_output["query_ts_feats"] = torch.stack([example["query_ts_feat"] for example in batch])
            batch_output["query_ts_masks"] = torch.stack([example["query_ts_mask"] for example in batch])
            batch_output["imputed_ts_feats"] = pad_sequence([example["imputed_ts_feat"] for example in batch], batch_first=True, padding_value=0)
            batch_output["imputed_ts_masks"] = pad_sequence([example["imputed_ts_mask"] for example in batch], batch_first=True, padding_value=0)
    except Exception:
        print("Sample with no vital signs detected")
        return

    if "text_feat" in batch[0].keys():
        text_embs = [pad_sequence(example["text_feat"], batch_first=True, padding_value=0) for example in batch]
        text_embs = pad_sequence(text_embs, batch_first=True, padding_value=0)
        note_times = pad_sequence([example["text_time"] for example in batch], batch_first=True, padding_value=0)
        note_time_masks = pad_sequence([example["text_mask"] for example in batch], batch_first=True, padding_value=0)

        batch_output["text_feats"] = text_embs
        batch_output["text_times"] = note_times
        batch_output["text_masks"] = note_time_masks

    if "cxr_feat" in batch[0].keys():
        cxr_feats = [pad_sequence(example["cxr_feat"], batch_first=True, padding_value=0) for example in batch]
        cxr_feats = pad_sequence(cxr_feats, batch_first=True, padding_value=0)
        cxr_times = pad_sequence([example["cxr_time"] for example in batch], batch_first=True, padding_value=0)
        cxr_time_masks = pad_sequence([example["cxr_mask"] for example in batch], batch_first=True, padding_value=0)

        batch_output["cxr_feats"] = cxr_feats
        batch_output["cxr_times"] = cxr_times
        batch_output["cxr_masks"] = cxr_time_masks

    if "ecg_feat" in batch[0].keys():
        ecg_feats = [pad_sequence(example["ecg_feat"], batch_first=True, padding_value=0) for example in batch]
        ecg_feats = pad_sequence(ecg_feats, batch_first=True, padding_value=0)
        ecg_times = pad_sequence([example["ecg_time"] for example in batch], batch_first=True, padding_value=0)
        ecg_time_masks = pad_sequence([example["ecg_mask"] for example in batch], batch_first=True, padding_value=0)

        batch_output["ecg_feats"] = ecg_feats
        batch_output["ecg_times"] = ecg_times
        batch_output["ecg_masks"] = ecg_time_masks

    return batch_output

