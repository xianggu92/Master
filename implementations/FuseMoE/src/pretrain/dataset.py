from utils.util import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
import os
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence


def data_perpare(args, mode, data=None):
    """
    Prepare the data for training or evaluation (Time Series only).
    """
    dataset = TS_IrgDataset(args, mode, data)

    if mode == 'train':
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.train_batch_size, collate_fn=TSIrgcollate_fn)
    else:
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size, collate_fn=TSIrgcollate_fn)

    return dataset, sampler, dataloader


class TS_IrgDataset(Dataset):
    """
    A PyTorch dataset class for handling time series data in the MIMIC-IV dataset.
    """
    def __init__(self, args, mode, data=None):
        if data is not None:
            self.data = data
        else:
            self.data = load_data(file_path=args.file_path, mode=mode, task=args.task)
            
        self.modeltype = args.modeltype
        self.mode = mode
        self.tt_max = args.tt_max
        
        if args.debug:
            self.data = self.data[:100]
        
    def __getitem__(self, idx):
        x = {}
        data_detail = self.data[idx]

        ts = data_detail['irg_ts']
        ts_tt = data_detail["ts_tt"]
        ts_mask = data_detail['irg_ts_mask']

        x = torch.tensor(ts, dtype=torch.float)
        mask = torch.tensor(ts_mask, dtype=torch.long)

        x['x'] = x
        x['mask'] = mask
        x['time'] = torch.tensor([t / self.tt_max for t in ts_tt], dtype=torch.float)

        # 初始化 label
        label = torch.zeros_like(x)
        
        # 找出所有真實觀測值（mask == 1）的索引
        valid_indices = torch.where(mask == 1)[0]
        
        if len(valid_indices) > 1:
            # 提取所有真實觀測值的數值
            valid_values = x[valid_indices]

            # 計算前後變化：下一個真實值 減去 當前真實值
            diff = valid_values[1:] - valid_values[:-1]
            
            # 將變化量填回對應的真實觀測值位置（最後一個真實觀測值沒有下一個，所以維持 0）
            label[valid_indices[:-1]] = diff
            
        x['label'] = label

        return x    

    def __len__(self):
        return len(self.data)


def load_data(file_path, mode, task='ihm'):
    """
    Load data from a file.
    """
    dataPath = os.path.join(file_path, mode + '_' + task + '_stays.pkl')
    data = []
    if os.path.isfile(dataPath):
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            data = pickle.load(f)
    return data


def TSIrgcollate_fn(batch):
    x = {}

    ts_input_sequences = pad_sequence([example['x'] for example in batch], batch_first=True, padding_value=0)
    ts_mask_sequences = pad_sequence([example['mask'] for example in batch], batch_first=True, padding_value=0)
    ts_tt = pad_sequence([example['time'] for example in batch], batch_first=True, padding_value=0)
    ts_label_sequences = pad_sequence([example['label'] for example in batch], batch_first=True, padding_value=0)

    x['x'] = ts_input_sequences
    x['mask'] = ts_mask_sequences
    x['time'] = ts_tt
    x['label'] = ts_label_sequences
        
    return x