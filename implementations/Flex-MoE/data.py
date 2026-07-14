import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import json
from models import Custom3DCNN, PatchEmbeddings
from torchvision.transforms import Compose, ToTensor, Normalize
import os
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from itertools import combinations
import os
import pickle
from torch.nn.utils.rnn import pad_sequence


class MultiModalDataset(Dataset):
    def __init__(self, data_dict, observed_idx, ids, labels, input_dims, transforms, masks, preprocessed=False, use_common_ids=True):
        self.data_dict = data_dict
        self.mc = np.array(data_dict['modality_comb'])
        self.observed = observed_idx
        self.ids = ids
        self.labels = labels
        self.input_dims = input_dims
        self.transforms = transforms
        self.masks = masks
        self.preprocessed = preprocessed
        self.use_common_ids = use_common_ids
        self.data_new = {modality: data[ids] for modality, data in self.data_dict.items() if 'modality' not in modality}
        self.label_new = self.labels[ids]
        self.mc_new = self.mc[ids]
        self.observed_new = self.observed[ids]

        # Sort ids by the number of available modalities
        self.sorted_ids = sorted(np.arange(len(ids)), key=lambda idx: sum([1 for modality in self.data_new if -2 not in self.data_new[modality][idx]]), reverse=True)
        self.data_new = {modality: data[self.sorted_ids] for modality, data in self.data_new.items()}
        self.label_new = self.label_new[self.sorted_ids]
        self.mc_new = self.mc_new[self.sorted_ids]
        self.observed_new = self.observed_new[self.sorted_ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_data = {}
        for modality, data in self.data_new.items():
            sample_data[modality] = data[idx]
            if (modality == 'image') & (not self.preprocessed):
                subj1 = data[idx]
                subj_gm_3d = np.zeros(self.masks.shape, dtype=np.float32)
                subj_gm_3d.ravel()[self.masks] = subj1
                subj_gm_3d = subj_gm_3d.reshape((91, 109, 91))
                if self.transforms:
                    subj_gm_3d = self.transforms(subj_gm_3d)
                sample = subj_gm_3d[None, :, :, :]  # Add channel dimension
                sample_data[modality] = np.array(sample)

        label = self.label_new[idx]
        mc = self.mc_new[idx]
        observed = self.observed_new[idx]

        return sample_data, label, mc, observed

def convert_ids_to_index(ids, index_map):
    return [index_map[id] if id in index_map else -1 for id in ids]

def load_and_preprocess_image_data(image_path, label_df, id_to_idx):
    # Load and preprocess image data
    image_data = np.load(os.path.join(image_path, 'ADNI_G.npy'), mmap_mode='r')
    mask_path = os.path.join(image_path, 'BLSA_SPGR+MPRAGE_averagetemplate_muse_seg_DS222.nii.gz')
    
    subject_ids = []
    dates = []
    with open('./data/adni/image/ADNI_subj.txt', 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split('_')
            subject_id = '_'.join(parts[:3])
            date = parts[-1]
            subject_ids.append(subject_id)
            dates.append(date)

    df = pd.DataFrame({
            'PTID': subject_ids,
            'date': dates
        })

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date', ascending=False)
    idx = df.groupby('PTID')['date'].idxmax()

    # Creating the subset DataFrame using the indexes
    subdf = df.loc[idx]
    subdf = subdf.sort_index()
    subdf = subdf.reset_index()

    merged_df = pd.merge(subdf, label_df, on='PTID', how='left')

    image_data = image_data[merged_df['index']]
    final_subject_ids = list(subdf.PTID)

    new_idx = np.array(convert_ids_to_index(final_subject_ids, id_to_idx))
    filtered_idx = [x for x in new_idx if x != -1]
    tmp = np.zeros((len(id_to_idx), image_data.shape[1])) - 2
    tmp[filtered_idx] = image_data[np.array(new_idx) != -1]

    data = nib.load(mask_path).get_fdata()
    mean = image_data.mean()
    std = image_data.std()     
    # mean = data.mean()
    # std = data.std()
    mask_gm = (data == 150).ravel()
    
    return tmp, filtered_idx, mean, std, mask_gm


def load_and_preprocess_data(args, modality_dict):
    # Paths
    image_path = './data/adni/image'
    preprocessed_image_path = './data/adni/image/UCSFFSX7_09Jun2025.csv'
    genomic_path = './data/adni/genomic/genomic_merged.h5ad'
    clinical_path = './data/adni/clinical/clinical_merged'
    biospecimen_path = './data/adni/biospecimen/biospecimen_merged'
    label_df = pd.read_csv('./data/adni/label.csv', index_col='PTID')
    label_df['DIAGNOSIS'] -= 1
    labels = label_df['DIAGNOSIS'].values.astype(np.int64)
    n_labels = len(set(labels))

    with open('./data/adni/PTID_splits.json') as json_file:
        data_split = json.load(json_file)

    train_ids = list(set(data_split['training']))
    valid_ids = list(set(data_split['validation']))
    test_ids = list(set(data_split['testing']))

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(label_df.index)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0],4), dtype=bool) # IGCB order

    # Initialize modality combination list
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if 'I' in args.modality or 'i' in args.modality:
        if args.preprocessed:
            df = pd.read_csv(preprocessed_image_path)
        
            # filter the latest record per subject using update_stamp
            df['update_stamp'] = pd.to_datetime(df['update_stamp'], errors='coerce')
            idx = df.groupby('PTID')['update_stamp'].idxmax()
            df = df.loc[idx].reset_index(drop=True)
            df.index = df['PTID']

            # select brain-related features ending with CV, TA, or SV.
            feature_cols = [col for col in df.columns if (
                col.endswith('CV') or col.endswith('TA') or col.endswith('SV')) and col.startswith('ST')
            ]
            df = df[feature_cols]

            if args.initial_filling == 'mean':
                df = df.apply(lambda x: x.fillna(x.mode().iloc[0]), axis=0)

            scaler = StandardScaler()
            brain_values = df.apply(pd.to_numeric, errors='coerce')  
            arr = scaler.fit_transform(brain_values.fillna(0)) 
            
            new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
            filtered_idx = new_idx[new_idx != -1]
            for idx in filtered_idx:
                update_modality_combinations(idx, 'I')
            tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
            tmp[filtered_idx] = arr[new_idx != -1]
            observed_idx_arr[filtered_idx, modality_dict['image']] = True
            data_dict['image'] = tmp.astype(np.float32)
            common_idx_list.append(set(filtered_idx))
            encoder_dict['image'] = PatchEmbeddings(df.shape[1], args.num_patches, args.hidden_dim).to(args.device)
            input_dims['image'] = df.shape[1]

        else:
            arr, filtered_idx, mean, std, mask = load_and_preprocess_image_data(image_path, label_df, id_to_idx)
            observed_idx_arr[:, modality_dict['image']] = arr[:, 0] != -2
            for idx in filtered_idx:
                update_modality_combinations(idx, 'I')

            data_dict['image'] = np.array(arr)
            common_idx_list.append(set(filtered_idx))
            encoder_dict['image'] = torch.nn.Sequential(
                Custom3DCNN(hidden_dim=args.hidden_dim).to(args.device),
                PatchEmbeddings(feature_size=args.hidden_dim, num_patches=args.num_patches, embed_dim=args.hidden_dim).to(args.device)
                )
            input_dims['image'] = arr.shape[1]
            transforms['image'] = Compose([
                                        ToTensor(),
                                        Normalize(mean=[mean], std=[std]),
                                    ])
            masks['image'] = mask

    if 'G' in args.modality or 'g' in args.modality:
        df = sc.read_h5ad(genomic_path).to_df()
        if args.initial_filling == 'mean':
            df = df.apply(lambda x: x.fillna(x.mode().iloc[0]), axis=0) # use mode as genotype values are 0,1,2
        arr = df.values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        arr = scaler.fit_transform(arr)
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['genomic']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'G')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]

        data_dict['genomic'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['genomic'] = PatchEmbeddings(df.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['genomic'] = df.shape[1]

    if 'C' in args.modality or 'c' in args.modality:
        if args.initial_filling == 'mean':
            path = clinical_path + '_mean.csv'
        else:
            path = clinical_path + '.csv'
        df = pd.read_csv(path, index_col=0)
        columns_to_exclude = [col for col in df.columns if col.startswith('PTCOGBEG') or col.startswith('PTADDX') or col.startswith('PTADBEG')]
        if len(columns_to_exclude) > 0:
            df = df.drop(columns_to_exclude, axis=1)
        arr = df.values.astype(np.float32)
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['clinical']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'C')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['clinical'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['clinical'] = PatchEmbeddings(df.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['clinical'] = df.shape[1]

    if 'B' in args.modality or 'b' in args.modality:
        if args.initial_filling == 'mean':
            path = biospecimen_path + '_mean.csv'
        else:
            path = biospecimen_path + '.csv'
        df = pd.read_csv(path, index_col=0)
        arr = df.values
        new_idx = np.array(convert_ids_to_index(df.index, id_to_idx))
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['biospecimen']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'B')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['biospecimen'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['biospecimen'] = PatchEmbeddings(df.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['biospecimen'] = df.shape[1]

    combination_to_index = get_modality_combinations(args.modality) # 0: full modality index
    modality_combinations = [''.join(sorted(set(comb))) for comb in modality_combinations]
    full_modality_index = min(list(combination_to_index.values()))
    assert (full_modality_index == 0) # max(list(combination_to_index.values()))
    _keys = combination_to_index.keys()
    data_dict['modality_comb'] = [combination_to_index[comb] if comb in _keys else -1 for comb in modality_combinations]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in valid_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    if args.use_common_ids:
        common_idxs = set.intersection(*common_idx_list)
        train_idxs = list(common_idxs & set(train_idxs))
        valid_idxs = list(common_idxs & set(valid_idxs))
        test_idxs = list(common_idxs & set(test_idxs))

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return all(data_dict[modality][idx, 0] == -2 for modality in data_dict.keys() if modality != 'modality_comb')

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]

    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index

def load_and_preprocess_data_mimic(args, modality_dict):
    # Paths
    lab_path = './data/mimic/lab_x'
    note_path = './data/mimic/note_x'
    code_path = './data/mimic/code_x'
    label_df = pd.read_csv('./data/mimic/labels.csv', index_col='subject_id')
    labels = label_df['one_year_mortality'].values.astype(np.int64)
    n_labels = len(set(labels))

    with open('./data/mimic/PTID_splits_mimic.json') as json_file:
        data_split = json.load(json_file)

    train_ids = list(set(data_split['training']))
    valid_ids = list(set(data_split['validation']))
    test_ids = list(set(data_split['testing']))

    data_dict = {}
    encoder_dict = {}
    input_dims = {}
    transforms = {}
    masks = {}

    id_to_idx = {id: idx for idx, id in enumerate(label_df.index)}
    common_idx_list = []
    observed_idx_arr = np.zeros((labels.shape[0], args.n_full_modalities), dtype=bool) # IGCB order

    # Initialize modality combination list
    modality_combinations = [''] * len(id_to_idx)

    def update_modality_combinations(idx, modality):
        nonlocal modality_combinations
        if modality_combinations[idx] == '':
            modality_combinations[idx] = modality
        else:
            modality_combinations[idx] += modality

    # Load modalities
    if 'L' in args.modality or 'l' in args.modality:
        path = lab_path
        arr = torch.load(path+'.pt')
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['lab']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'L')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['lab'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['lab'] = PatchEmbeddings(arr.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['lab'] = arr.shape[1]

    if 'N' in args.modality or 'n' in args.modality:
        path = note_path
        arr = torch.load(path+'.pt')
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['note']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'N')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['note'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['note'] = PatchEmbeddings(arr.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['note'] = arr.shape[1]

    if 'C' in args.modality or 'c' in args.modality:
        path = code_path
        arr = torch.load(path+'.pt')
        new_idx = np.arange(arr.shape[0])
        filtered_idx = new_idx[new_idx != -1]
        observed_idx_arr[filtered_idx, modality_dict['code']] = True
        for idx in filtered_idx:
            update_modality_combinations(idx, 'C')
        tmp = np.zeros((len(id_to_idx), arr.shape[1])) - 2
        tmp[filtered_idx] = arr[new_idx != -1]
        
        data_dict['code'] = tmp.astype(np.float32)
        common_idx_list.append(set(filtered_idx))
        encoder_dict['code'] = PatchEmbeddings(arr.shape[1], args.num_patches, args.hidden_dim).to(args.device)
        input_dims['code'] = arr.shape[1]
    
    combination_to_index = get_modality_combinations(args.modality) # 0: full modality index
    modality_combinations = [''.join(sorted(set(comb))) for comb in modality_combinations]
    full_modality_index = min(list(combination_to_index.values()))
    assert (full_modality_index == 0) # max(list(combination_to_index.values()))
    _keys = combination_to_index.keys()
    data_dict['modality_comb'] = [combination_to_index[comb] if comb in _keys else -1 for comb in modality_combinations]

    train_idxs = [id_to_idx[id] for id in train_ids if id in id_to_idx]
    valid_idxs = [id_to_idx[id] for id in valid_ids if id in id_to_idx]
    test_idxs = [id_to_idx[id] for id in test_ids if id in id_to_idx]

    if args.use_common_ids:
        common_idxs = set.intersection(*common_idx_list)
        train_idxs = list(common_idxs & set(train_idxs))
        valid_idxs = list(common_idxs & set(valid_idxs))
        test_idxs = list(common_idxs & set(test_idxs))

    # Remove rows where all modalities are missing (-2)
    def all_modalities_missing(idx):
        return all(data_dict[modality][idx, 0] == -2 for modality in data_dict.keys() if modality != 'modality_comb')

    train_idxs = [idx for idx in train_idxs if not all_modalities_missing(idx)]

    return data_dict, encoder_dict, labels, train_idxs, valid_idxs, test_idxs, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index

def collate_fn(batch):
    data, labels, mcs, observeds = zip(*batch)
    modalities = data[0].keys()
    collated_data = {modality: torch.tensor(np.stack([d[modality] for d in data]), dtype=torch.float32) for modality in modalities}
    labels = torch.tensor(labels, dtype=torch.long)
    mcs = torch.tensor(mcs, dtype=torch.long)
    observeds = torch.tensor(np.vstack(observeds))
    return collated_data, labels, mcs, observeds

def create_loaders(data_dict, observed_idx, labels, train_ids, valid_ids, test_ids, batch_size, num_workers, pin_memory, input_dims, transforms, masks, preprocessed, use_common_ids=True):
    if ('image' in list(data_dict.keys())) & (not preprocessed):
        train_transfrom = val_transform = test_transform = transforms['image']
        # val_transform = test_transform = False
        mask = masks['image']
    else:
        train_transfrom = val_transform = test_transform = False
        mask = None

    train_dataset = MultiModalDataset(data_dict, observed_idx, train_ids, labels, input_dims, train_transfrom, mask, preprocessed, use_common_ids)
    valid_dataset = MultiModalDataset(data_dict, observed_idx, valid_ids, labels, input_dims, val_transform, mask, preprocessed, use_common_ids)
    test_dataset = MultiModalDataset(data_dict, observed_idx, test_ids, labels, input_dims, test_transform, mask, preprocessed, use_common_ids)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    train_loader_shuffle = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, train_loader_shuffle, val_loader, test_loader

# Updated: full modality index is 0.
def get_modality_combinations(modalities):
    all_combinations = []
    for i in range(len(modalities), 0, -1):
        comb = list(combinations(modalities, i))
        all_combinations.extend(comb)
    
    # Create a mapping dictionary
    combination_to_index = {''.join(sorted(comb)): idx for idx, comb in enumerate(all_combinations)}
    return combination_to_index

def data_prepare(args, mode, data=None):
    """Prepare the data for training or evaluation."""
    dataset = TSNoteIrgDataset(args, mode, data)

    if mode == "train":
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=text_ts_irg_collate_fn)
        sampler = RandomSampler(dataset)
        dataloader_shuffle = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=text_ts_irg_collate_fn)
    else:
        dataloader = None
        sampler = SequentialSampler(dataset)
        dataloader_shuffle = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=text_ts_irg_collate_fn)

    encoder_dict = None
    if mode == 'train':
        max_time = 48 if 'ihm' or 'los' in args.task else 24
        encoder_dict = {}

        encoder_dict['ts_feats'] = PatchEmbeddings(max_time * 30, args.num_patches, args.hidden_dim).to(args.device)
        if 'I' in args.modality:
            encoder_dict['cxr_feats'] = PatchEmbeddings(1024, args.num_patches, args.hidden_dim).to(args.device)
        if 'T' in args.modality:
            encoder_dict['text_feats'] = PatchEmbeddings(768, args.num_patches, args.hidden_dim).to(args.device)
        if 'E' in args.modality:
            encoder_dict['ecg_feats'] = PatchEmbeddings(256, args.num_patches, args.hidden_dim).to(args.device)

    return dataset, sampler, dataloader, dataloader_shuffle, encoder_dict


def impute_missing_values(features, timestamps, feature_mask, duration, max_time):
    """Imputes missing values in the input data based on the discretization rule mentioned in the paper."""
    num_features = features.shape[1]
    imputed_data = np.zeros(shape=(max_time // duration, num_features))

    for feat_row, time_val, mask_row in zip(features, timestamps, feature_mask):
        target_row_idx = int(time_val / duration)
        if target_row_idx >= max_time:
            continue

        for feat_idx, (feat_val, is_present) in enumerate(zip(feat_row, mask_row)):
            if is_present == 1:
                imputed_data[target_row_idx][feat_idx] = feat_val
            else:
                if imputed_data[target_row_idx - 1][feat_idx] != 0:
                    imputed_data[target_row_idx][feat_idx] = imputed_data[target_row_idx - 1][feat_idx]

    return imputed_data


class TSNoteIrgDataset(Dataset):
    """A PyTorch dataset class for handling time series note data in the MIMIC-IV dataset."""

    def __init__(self, args, mode, data=None):
        if data is not None:
            self.data = data
        else:
            self.data = load_data(file_path=args.file_path, mode=mode, task=args.task)
        self.args = args
        self.model_type = args.modality
        self.mode = mode

        if 'ihm' or 'los' in args.task:
            self.max_time = 48
        else:
            self.max_time = 24

        observed_idx_arr = np.zeros((len(self.data), len(self.model_type)), dtype=bool)

        modality_cols = {
            "timeseries": None,          # 時間序列（無缺失欄位，預設全存在）
            "text": "text_missing",      # 文字
            "cxr": "cxr_missing",        # 圖片
            "ecg": "ecg_missing"         # ECG
        }

        for col_idx, (modality, missing_col) in enumerate(modality_cols.items()):
            if missing_col is None:
                observed_idx_arr[:, col_idx] = True
            else:
                missing_values = np.array([bool(item[missing_col]) for item in self.data], dtype=bool)
                observed_idx_arr[:, col_idx] = ~missing_values

        # 依模態數量由多到少排序
        modality_counts = observed_idx_arr.sum(axis=1)
        sorted_ids = np.argsort(modality_counts)[::-1]
        self.data = [self.data[idx] for idx in sorted_ids]
        self.observed_idx_arr = observed_idx_arr[sorted_ids]

        if args.debug:
            self.data = self.data[:100]

    def __getitem__(self, idx):
        sample_dict = {}
        data_detail = self.data[idx]

        label = torch.tensor(data_detail["label"], dtype=torch.long)
        sample_dict["label"] = label

        if "T" in self.model_type:
            ts_features = data_detail["irg_ts"]
            ts_timestamps = data_detail["ts_tt"].astype(np.float32)
            ts_mask = data_detail["irg_ts_mask"]

            regularized_ts = impute_missing_values(ts_features, ts_timestamps, ts_mask, 1, self.max_time)
            regularized_ts = torch.tensor(regularized_ts, dtype=torch.float).T.flatten()

            sample_dict["reg_ts_feat"] = regularized_ts

        if "N" in self.model_type:
            if not data_detail["text_missing"]:
                text_embeddings = np.mean(data_detail["text_embeddings"], axis=0)
                text_embeddings = torch.tensor(text_embeddings, dtype=torch.float)
            else:
                text_embeddings = torch.zeros((1, 768))

            sample_dict["text_feat"] = text_embeddings
            sample_dict["text_missing"] = data_detail["text_missing"]

        if "I" in self.model_type:
            if not data_detail["cxr_missing"]:
                cxr_feats = np.mean(data_detail["cxr_feats"], axis=0)
                cxr_feats = torch.tensor(cxr_feats, dtype=torch.float)
            else:
                cxr_feats = torch.zeros((1, 1024))

            sample_dict["cxr_feat"] = cxr_feats
            sample_dict["cxr_missing"] = data_detail["cxr_missing"]

        if "E" in self.model_type:
            if not data_detail["ecg_missing"]:
                ecg_feats = np.mean(data_detail["ecg_feats"], axis=0)
                ecg_feats = torch.tensor(ecg_feats, dtype=torch.float)
            else:
                ecg_feats = torch.zeros((1, 256))

            sample_dict["ecg_feat"] = ecg_feats
            sample_dict["ecg_missing"] = data_detail["ecg_missing"]

        sample_dict['observed_idx_arr'] = torch.tensor(self.observed_idx_arr[idx], dtype=torch.bool)

        binary_arr = (~self.observed_idx_arr[idx]).astype(int)
        weights = np.array([8, 4, 2, 1])
        sample_dict['modality_combination'] = torch.tensor(np.dot(binary_arr, weights), dtype=torch.long)

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
    batch_samples = {}
    batch_labels = torch.stack([sample["label"] for sample in batch])

    try:
        batch_samples["ts_feats"] = torch.stack([sample["reg_ts_feat"] for sample in batch])
    except Exception:
        print("Sample with no vital signs detected")
        return

    if "text_feat" in batch[0].keys():
        batch_samples["text_feats"] = torch.stack([sample["text_feat"] for sample in batch])

    if "cxr_feat" in batch[0].keys():
        batch_samples["cxr_feats"] = torch.stack([sample["cxr_feat"] for sample in batch])

    if "ecg_feat" in batch[0].keys():
        batch_samples["ecg_feats"] = torch.stack([sample["ecg_feat"] for sample in batch])

    batch_mcs = torch.stack([sample["modality_combination"] for sample in batch])
    batch_observed = torch.stack([sample["observed_idx_arr"] for sample in batch])

    return batch_samples, batch_labels, batch_mcs, batch_observed