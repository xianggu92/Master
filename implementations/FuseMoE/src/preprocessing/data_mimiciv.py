import os
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd


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


def get_tau(ts_tt, ts_mask):
    # L, [L, K]
    tmp_time = ts_mask * np.expand_dims(ts_tt, axis=-1)  # [L,K]

    new_mask = ts_mask.copy()
    new_mask[0, :] = 1
    tmp_time[new_mask == 0] = np.nan

    # padding the missing value with the next value
    df1 = pd.DataFrame(tmp_time)
    df1 = df1.fillna(method='ffill')
    tmp_time = np.array(df1)

    tmp_time[1:, :] -= tmp_time[:-1, :]
    del new_mask
    return tmp_time * ts_mask


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
        if args.debug:
            self.data = self.data[:100]

    def __getitem__(self, idx):
        sample_dict = {}
        data_detail = self.data[idx]

        label = torch.tensor(data_detail["label"], dtype=torch.long)
        sample_dict["label"] = label

        if "TS" in self.model_type:
            ts_features = data_detail["irg_ts"]
            ts_timestamps = data_detail["ts_tt"].astype(np.float32)
            ts_mask = data_detail["irg_ts_mask"]

            if self.reg_ts:
                regularized_ts = impute_missing_values(ts_features, ts_timestamps, ts_mask, 1, self.max_time)
                regularized_ts = torch.tensor(regularized_ts, dtype=torch.float)
            else:
                regularized_ts = None

            ts_tau = torch.tensor(get_tau(ts_timestamps, ts_mask), dtype=torch.float)
            ts_features = torch.tensor(ts_features, dtype=torch.float)
            ts_mask = torch.tensor(ts_mask, dtype=torch.float)
            ts_timestamps = torch.tensor(ts_timestamps / self.max_time, dtype=torch.float)

            sample_dict["ts_feat"] = ts_features
            sample_dict["ts_time"] = ts_timestamps
            sample_dict["ts_mask"] = ts_mask
            sample_dict["reg_ts_feat"] = regularized_ts
            sample_dict["ts_tau"] = ts_tau

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
                ecg_feats = torch.tensor(np.array(ecg_feats), dtype=torch.float)

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
        ts_taus = pad_sequence([example["ts_tau"] for example in batch], batch_first=True, padding_value=0)
        labels = torch.stack([example["label"] for example in batch])

        if batch[0]["reg_ts_feat"] is not None:
            reg_ts_input = torch.stack([example["reg_ts_feat"] for example in batch])
        else:
            reg_ts_input = None

        batch_output["ts_feats"] = ts_input_sequences
        batch_output["ts_masks"] = ts_mask_sequences
        batch_output["ts_times"] = ts_timestamps
        batch_output["reg_ts_feats"] = reg_ts_input
        batch_output["ts_taus"] = ts_taus
        batch_output["labels"] = labels
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