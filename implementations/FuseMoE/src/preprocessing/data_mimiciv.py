from utils.util import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import os
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence
import pdb


def data_perpare(args, mode, data=None):
    """
    Prepare the data for training or evaluation.

    Args:
        args (object): The arguments object.
        mode (str): The mode, either 'train' or 'eval'.
        tokenizer (object): The tokenizer object.
        data (list, optional): The data to be used. Defaults to None.

    Returns:
        dataset (object): The dataset object.
        sampler (object): The sampler object.
        dataloader (object): The dataloader object.
    """
    dataset = TSNote_Irg(args, mode, data)

    if mode=='train':
        sampler = RandomSampler(dataset)
        dataloader= DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, collate_fn=TextTSIrgcollate_fn)
    else:
        sampler = SequentialSampler(dataset)
        dataloader= DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, collate_fn=TextTSIrgcollate_fn)

    return dataset, sampler, dataloader


def F_impute(X, tt, mask, duration, tt_max):
    """
    Imputes missing values in the input data based on the discretization rule mentioned in the paper.

    Parameters:
    X (numpy.ndarray): Input data matrix of shape (n_samples, n_features).
    tt (numpy.ndarray): Array of time values corresponding to each sample.
    mask (numpy.ndarray): Array indicating missing values in the input data.
    duration (int): Duration of each time interval for discretization.
    tt_max (int): Maximum time value.

    Returns:
    numpy.ndarray: Imputed data matrix of shape (tt_max//duration, n_features*2).
    """
    
    no_feature = X.shape[1]
    impute = np.zeros(shape=(tt_max//duration,no_feature*2))
    for x, t, m in zip(X, tt, mask):
        row=int(t/duration)
        if row>=tt_max:
            continue
        for f_idx, (row_x, row_m) in enumerate(zip(x, m)):
            # perform imputation according to the discretization rule in paper
            if row_m == 1:
                impute[row][no_feature+f_idx] = 1
                impute[row][f_idx] = row_x
            else:
                if impute[row-1][f_idx] != 0:
                    impute[row][f_idx] = impute[row-1][f_idx]

    return impute


class TSNote_Irg(Dataset):
    """
    A PyTorch dataset class for handling time series note data in the MIMIC-IV dataset.

    Args:
        args (argparse.Namespace): The command-line arguments.
        mode (str): The mode of the dataset (e.g., "train", "val", "test").
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding the text data.
        data (list, optional): The list of data samples. If not provided, the data will be loaded from a file.

    Attributes:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for encoding the text data.
        max_len (int): The maximum length of the input sequences.
        data (list): The list of data samples.
        chunk (bool): Whether to chunk the data.
        text_id_attn_data (list): The list of text data samples for attention calculation.
        padding (str): The padding strategy for the input sequences.
        notes_order (str): The order of the notes.
        order_sample (numpy.ndarray): The array of randomly sampled note orders.
        modeltype (str): The type of the model.
        model_name (str): The name of the model.
        num_of_notes (int): The number of notes to consider.
        tt_max (float): The maximum value of the time-to-end feature.

    Methods:
        __getitem__(self, idx): Retrieves the data at the given index.
        __len__(self): Returns the length of the dataset.
    """
    
    def __init__(self, args, mode, data=None):
        if data != None:
            self.data = data
        else:
            self.data = load_data(file_path=args.file_path, mode=mode, task=args.task)
        self.modeltype = args.modeltype
        self.mode = mode
        self.tt_max = args.tt_max
        self.reg_ts = args.reg_ts
        
    def __getitem__(self, idx):
        """
        Retrieves the data at the given index.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            dict: A dictionary containing the data at the given index.
        """
        x = {}
        data_detail = self.data[idx]

        label = data_detail["label"]
        label = torch.tensor(label, dtype=torch.long)
        x['label'] = label

        if 'TS' in self.modeltype:
            ts = data_detail['irg_ts']
            ts_tt = data_detail["ts_tt"]
            ts_mask = data_detail['irg_ts_mask']

            # 前處理 ipynb 檔沒有插值到最大的時間點，因此要重新處理一次插值後的 time series
            if self.reg_ts:
                reg_ts = F_impute(ts, ts_tt, ts_mask, 1, self.tt_max)
                reg_ts = torch.tensor(reg_ts, dtype=torch.float)
            else:
                reg_ts = None

            # 在 F.impute 之後才能轉換成 tensor，不然處理速度會很慢
            ts = torch.tensor(ts, dtype=torch.float)
            ts_mask = torch.tensor(ts_mask, dtype=torch.long)
            ts_tt = torch.tensor([t/self.tt_max for t in ts_tt], dtype=torch.float)

            x['ts'] = ts
            x['ts_tt'] = ts_tt
            x['ts_mask'] = ts_mask
            x['reg_ts'] = reg_ts

        if 'Text' in self.modeltype:
            if not data_detail['text_missing']:
                text_emb = data_detail['text_embeddings']
                text_emb = torch.tensor(text_emb, dtype=torch.float)

                text_time_to_end = data_detail["text_time_to_end"]
                text_time_to_end = [1-t/self.tt_max for t in text_time_to_end]
                text_time_to_end = torch.tensor(text_time_to_end, dtype=torch.float)

                text_time_mask = [1] * len(text_time_to_end)
                text_time_mask = torch.tensor(text_time_mask, dtype=torch.long)
            else:
                text_emb = torch.zeros((1, 768))
                text_time_to_end = torch.zeros(1)
                text_time_mask = torch.ones(1)

            x['note_feats'] = text_emb
            x['note_time'] = text_time_to_end
            x['note_time_mask'] = text_time_mask
            x['text_missing'] = data_detail['text_missing']
            x['text_data'] = data_detail['text_data']

        if 'CXR' in self.modeltype:
            if not data_detail['cxr_missing']:
                cxr_feats = data_detail['cxr_feats']
                cxr_feats = torch.tensor(cxr_feats, dtype=torch.float)

                cxr_time_to_end = data_detail['cxr_time'].astype(np.float32)
                cxr_time_to_end = torch.tensor(cxr_time_to_end, dtype=torch.float)

                cxr_time_mask = [1] * len(cxr_time_to_end)
                cxr_time_mask = torch.tensor(cxr_time_mask, dtype=torch.long)
            else:
                cxr_feats = torch.zeros((1, 1024))
                cxr_time_to_end = torch.zeros(1)
                cxr_time_mask = torch.ones(1)

            x['cxr_feats'] = cxr_feats
            x['cxr_time'] = cxr_time_to_end
            x['cxr_time_mask'] = cxr_time_mask
            x['cxr_missing'] = data_detail['cxr_missing']

        if 'ECG' in self.modeltype:
            if not data_detail['ecg_missing']:
                ecg_feats = data_detail['ecg_feats']
                ecg_feats = torch.tensor(ecg_feats, dtype=torch.float)

                # If any ecg_feats are nan, replace with 0
                ecg_feats[torch.isnan(ecg_feats)] = 0

                # If any ecg_feats are inf, replace with 0
                ecg_feats[torch.isinf(ecg_feats)] = 0

                ecg_time_to_end = data_detail['ecg_time'].astype(np.float32)
                ecg_time_to_end = torch.tensor(ecg_time_to_end, dtype=torch.float)

                ecg_time_mask = [1] * len(ecg_time_to_end)
                ecg_time_mask = torch.tensor(ecg_time_mask, dtype=torch.long)
            else:
                ecg_feats = torch.zeros((1, 256))
                ecg_time_to_end = torch.zeros(1)
                ecg_time_mask = torch.ones(1)

            x['ecg_feats'] = ecg_feats
            x['ecg_time'] = ecg_time_to_end
            x['ecg_time_mask'] = ecg_time_mask
            x['ecg_missing'] = data_detail['ecg_missing']

        return x    

    def __len__(self):
        return len(self.data)

def load_data(file_path, mode, text=False, task='ihm'):
    """
    Load data from a file.

    Args:
        file_path (str): The path to the file.
        mode (str): The mode of the data.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        text (bool, optional): Whether the data is text. Defaults to False.
        task (str, optional): The task of the data. Defaults to 'ihm'.

    Returns:
        data: The loaded data.
    """
    dataPath = os.path.join(file_path, mode + '_' + task + '_stays.pkl')
    if os.path.isfile(dataPath):
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            data = pickle.load(f)

    return data

def TextTSIrgcollate_fn(batch):
    x = {}

    if 'text_missing' in batch[0].keys():
        text_missing = torch.stack([torch.tensor(example["text_missing"]) for example in batch])
        x['text_missing'] = text_missing

    if 'cxr_missing' in batch[0].keys():
        cxr_missing = torch.stack([torch.tensor(example["cxr_missing"]) for example in batch])
        x['cxr_missing'] = cxr_missing

    if 'ecg_missing' in batch[0].keys():
        ecg_missing = torch.stack([torch.tensor(example["ecg_missing"]) for example in batch])
        x['ecg_missing'] = ecg_missing

    try:
        ts_input_sequences = pad_sequence([example['ts'] for example in batch], batch_first=True, padding_value=0)
        ts_mask_sequences = pad_sequence([example['ts_mask'] for example in batch], batch_first=True, padding_value=0)
        ts_tt = pad_sequence([example['ts_tt'] for example in batch], batch_first=True, padding_value=0 )
        labels = torch.stack([example["label"] for example in batch])
        
        if batch[0]['reg_ts'] is not None:
            reg_ts_input=torch.stack([example['reg_ts'] for example in batch])
        else:
            reg_ts_input=None

        x['x_ts'] = ts_input_sequences
        x['x_ts_mask'] = ts_mask_sequences
        x['ts_tt_list'] = ts_tt
        x['reg_ts'] = reg_ts_input
        x['labels'] = labels
    except:
        # if there is no vital signs, just return
        print('Sample with no vital signs detected')
        return

    if 'note_feats' in batch[0].keys():
        text_emb = [pad_sequence(example['note_feats'], batch_first=True, padding_value=0) for example in batch]
        text_emb = pad_sequence(text_emb, batch_first=True, padding_value=0)
        note_time = pad_sequence([example['note_time'] for example in batch], batch_first=True, padding_value=0)
        note_time_mask = pad_sequence([example['note_time_mask'] for example in batch], batch_first=True, padding_value=0)
        text_data = [example['text_data'] for example in batch]

        x['text_emb'] = text_emb
        x['note_time_list'] = note_time
        x['note_time_mask_list'] = note_time_mask
        x['text_data'] = text_data

    if 'cxr_feats' in batch[0].keys():
        cxr_feats = [pad_sequence(example['cxr_feats'], batch_first=True, padding_value=0) for example in batch]
        cxr_feats = pad_sequence(cxr_feats, batch_first=True, padding_value=0)
        cxr_time = pad_sequence([example['cxr_time'] for example in batch], batch_first=True, padding_value=0)
        cxr_time_mask = pad_sequence([example['cxr_time_mask'] for example in batch], batch_first=True, padding_value=0)

        x['cxr_feats'] = cxr_feats
        x['cxr_time'] = cxr_time
        x['cxr_time_mask'] = cxr_time_mask

    if 'ecg_feats' in batch[0].keys():
        ecg_feats = [pad_sequence(example['ecg_feats'], batch_first=True, padding_value=0) for example in batch]
        ecg_feats = pad_sequence(ecg_feats, batch_first=True, padding_value=0)
        ecg_time = pad_sequence([example['ecg_time'] for example in batch], batch_first=True, padding_value=0)
        ecg_time_mask = pad_sequence([example['ecg_time_mask'] for example in batch], batch_first=True, padding_value=0)

        x['ecg_feats'] = ecg_feats
        x['ecg_time'] = ecg_time
        x['ecg_time_mask'] = ecg_time_mask

    return x