import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
import sys
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


# 查看是否有可用的 GPU 列表
# gpus = tf.config.list_physical_devices('GPU')
# print("可用的 GPU 數量: ", len(gpus))
# print("GPU 詳細資訊: ", gpus)

mimic_iv_path = "/mnt/nfs_share/Public_Data/Dataset_MIMICs/physionet.org/files/mimic-iv/2.2/"
mm_dir = "/mnt/data/yihua/master/datasets/mimic-iv"

output_dir = os.path.join(mm_dir, "preprocessing")
os.makedirs(output_dir, exist_ok=True)


f_path = os.path.join(mimic_iv_path, "hosp", "admissions.csv")
admissions_df = pd.read_csv(f_path, low_memory=False)
admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
admissions_df['dischtime'] = pd.to_datetime(admissions_df['dischtime'])

icustays_df = pd.read_csv(os.path.join(mimic_iv_path, "icu", "icustays.csv"), low_memory=False)
icustays_df['intime'] = pd.to_datetime(icustays_df['intime'])
icustays_df['outtime'] = pd.to_datetime(icustays_df['outtime'])


ecg_folder = '/mnt/nfs_share/Public_Data/Dataset_MIMICs/physionet.org/files/mimic-iv-ecg/1.0/'

records_list_df = pd.read_csv(os.path.join(ecg_folder, 'record_list.csv'))
records_list_df['ecg_time'] = pd.to_datetime(records_list_df['ecg_time'])


def calc_time_delta_hrs(icu_intime, charttime):
    return (charttime - icu_intime).total_seconds() / 3600


print('開始讀取 ECG 病患資料')
out_df = pd.DataFrame()
for index, row in tqdm(icustays_df.iterrows(), total=icustays_df.shape[0]):
    curr_subject_no = row['subject_id']
    curr_hadm_id = row['hadm_id']
    curr_stay_id = row['stay_id']
    curr_intime = row['intime']
    curr_outtime = row['outtime']

    # Check if subject has ECG data
    curr_subject_ecg = records_list_df[records_list_df['subject_id'] == curr_subject_no]
    curr_subject_ecg = curr_subject_ecg[curr_subject_ecg['ecg_time'] >= curr_intime]
    curr_subject_ecg = curr_subject_ecg[curr_subject_ecg['ecg_time'] <= curr_outtime]

    if curr_subject_ecg.shape[0] == 0:
        continue

    for ecg_index, ecg_row in curr_subject_ecg.iterrows():
        tmp_dict = {'subject_id': curr_subject_no,
                    'hadm_id': curr_hadm_id,
                    'stay_id': curr_stay_id,
                    'icu_time_delta': calc_time_delta_hrs(curr_intime, ecg_row['ecg_time']),
                    'ecg_time': ecg_row['ecg_time'],
                    'path': ecg_row['path']}
        tmp_df = pd.DataFrame(tmp_dict, index=[0])
        out_df = pd.concat([out_df, tmp_df], axis=0, ignore_index=True)

# tensorflow==2.15.0
f_path = '/mnt/data/yihua/master/implementations/FuseMoE/attia_encoder_256.keras'
encoder = tf.keras.models.load_model(f_path)

def load_ecg(path, stop_index=4096):
    rd_record = wfdb.rdrecord(path) 
    sig = rd_record.p_signal
    sig = sig[:stop_index, :]
    return sig

out_df['embeddings'] = None

for index, row in tqdm(out_df.iterrows(), total=out_df.shape[0]):
    curr_ecg_path = os.path.join(ecg_folder, row['path'])
    wf = load_ecg(curr_ecg_path)
    out_df.at[index, 'embeddings'] = encoder.predict(wf.reshape(1, -1, 12), verbose=0)


out_df.to_pickle(os.path.join(output_dir, "ecg_embeddings_icu.pkl"))


