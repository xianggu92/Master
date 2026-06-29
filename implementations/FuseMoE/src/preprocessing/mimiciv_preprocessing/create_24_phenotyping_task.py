import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pickle


mimic_iv_path = "/mnt/nfs_share/Public_Data/Dataset_MIMICs/physionet.org/files/mimic-iv/2.2/"
mm_dir = "/mnt/data/yihua/master/datasets/mimic-iv"

output_dir = os.path.join(mm_dir, "preprocessing")


include_notes = True
include_cxr = False
include_ecg = False
standard_scale = True
include_missing = False
restrict_24_hours = True


# ireg_vitals_ts_df = pd.read_pickle(os.path.join(output_dir, "ts_vitals_icu.pkl"))
# imputed_vitals = pd.read_pickle(os.path.join(output_dir, "imputed_ts_vitals_icu.pkl"))

ireg_vitals_ts_df = pd.read_pickle(os.path.join(output_dir, "ts_labs_vitals_icu.pkl"))
imputed_vitals = pd.read_pickle(os.path.join(output_dir, "imputed_ts_labs_vitals_icu.pkl"))

ireg_vitals_ts_df = ireg_vitals_ts_df[ireg_vitals_ts_df['timedelta'] >= 0]
imputed_vitals = imputed_vitals[imputed_vitals['timedelta'] >= 0]

if restrict_24_hours:
    ireg_vitals_ts_df = ireg_vitals_ts_df[ireg_vitals_ts_df['timedelta'] <= 24]
    imputed_vitals = imputed_vitals[imputed_vitals['timedelta'] <= 24]


if include_notes:
    notes_df = pd.read_pickle(os.path.join(output_dir, "icu_notes_text_embeddings.pkl"))
    notes_df = notes_df[notes_df['stay_id'].notnull()]

    notes_df = notes_df[notes_df['icu_time_delta'] >= 0]
    if restrict_24_hours:
        notes_df = notes_df[notes_df['icu_time_delta'] <= 24]

if include_cxr:
    cxr_df = pd.read_pickle(os.path.join(output_dir, "cxr_embeddings_icu.pkl"))

    cxr_df = cxr_df[cxr_df['icu_time_delta'] >= 0]
    if restrict_24_hours:
        cxr_df = cxr_df[cxr_df['icu_time_delta'] <= 24]

if include_ecg:
    ecg_df = pd.read_pickle(os.path.join(output_dir, "ecg_embeddings_icu.pkl"))
    ecg_df = ecg_df[ecg_df['icu_time_delta'] >= 0]
    if restrict_24_hours:
        ecg_df = ecg_df[ecg_df['icu_time_delta'] <= 24]


icustays_df = pd.read_csv(os.path.join(mimic_iv_path, "icu", "icustays.csv"), low_memory=False)
icustays_df['intime'] = pd.to_datetime(icustays_df['intime'])
icustays_df['outtime'] = pd.to_datetime(icustays_df['outtime'])


valid_stay_ids = icustays_df['stay_id'].unique()

ireg_vitals_ts_df = ireg_vitals_ts_df[ireg_vitals_ts_df['stay_id'].isin(valid_stay_ids)]
imputed_vitals = imputed_vitals[imputed_vitals['stay_id'].isin(valid_stay_ids)]

if include_notes:
    notes_df = notes_df[notes_df['stay_id'].isin(valid_stay_ids)]

if include_cxr:
    cxr_df = cxr_df[cxr_df['stay_id'].isin(valid_stay_ids)]

if include_ecg:
    ecg_df = ecg_df[ecg_df['stay_id'].isin(valid_stay_ids)]    



if not include_missing:
    unique_stays = ireg_vitals_ts_df['stay_id'].unique()
    print(f"Number of stays with vitals: {len(unique_stays)}")

    if include_notes:
        unique_stays = np.intersect1d(unique_stays, notes_df['stay_id'].unique())
        print(f"Number of stays with notes: {len(unique_stays)}")

    if include_cxr:
        unique_stays = np.intersect1d(unique_stays, cxr_df['stay_id'].unique())
        print(f"Number of stays with cxr: {len(unique_stays)}")

    if include_ecg:
        unique_stays = np.intersect1d(unique_stays, ecg_df['stay_id'].unique())
        print(f"Number of stays with ecg: {len(unique_stays)}")        
else:
    unique_stays = ireg_vitals_ts_df['stay_id'].unique()
    print(f"Number of stays with vitals: {len(unique_stays)}")

    if include_notes:
        # Get stays with either TS or notes
        unique_stays = np.union1d(unique_stays, notes_df['stay_id'].unique())
        print(f"Number of stays with either TS or notes: {len(unique_stays)}")
    
    if include_cxr:
        unique_stays = np.union1d(unique_stays, cxr_df['stay_id'].unique())
        print(f"Number of stays with either TS, notes, cxr: {len(unique_stays)}")

    if include_ecg:
        unique_stays = np.union1d(unique_stays, ecg_df['stay_id'].unique())
        print(f"Number of stays with either TS, notes, cxr, ecg: {len(unique_stays)}")        


# Create train, val, test splits
np.random.seed(0)
np.random.shuffle(unique_stays)
train_stays = unique_stays[:int(0.7*len(unique_stays))]
val_stays = unique_stays[int(0.7*len(unique_stays)):int(0.85*len(unique_stays))]
test_stays = unique_stays[int(0.85*len(unique_stays)):]


train_ireg_ts_df = ireg_vitals_ts_df[ireg_vitals_ts_df['stay_id'].isin(train_stays)].copy()
train_imputed_df = imputed_vitals[imputed_vitals['stay_id'].isin(train_stays)].copy()

cols = train_ireg_ts_df.columns.tolist()
cols = [col for col in cols if col not in ['subject_id', 'hadm_id', 'stay_id', 'timedelta']]

if standard_scale:
    scalers = {}

    for col in cols:
        # 計算第 25 百分位數 (Q1) 與 第 75 百分位數 (Q3)
        q1 = train_ireg_ts_df[col].quantile(0.25)
        q3 = train_ireg_ts_df[col].quantile(0.75)

        # 計算 IQR
        iqr = q3 - q1

        # 定義合理範圍的上下界
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        # 使用 pandas 的 clip 函數進行裁剪
        ireg_vitals_ts_df[col] = ireg_vitals_ts_df[col].clip(lower=lower_bound, upper=upper_bound)
        
        scaler = StandardScaler()
        scaler.fit(train_ireg_ts_df[[col]])
        ireg_vitals_ts_df[col] = scaler.transform(ireg_vitals_ts_df[[col]])
        scalers[col] = scaler

        scaler = StandardScaler()
        scaler.fit(train_imputed_df[[col]])
        imputed_vitals[col] = scaler.transform(imputed_vitals[[col]])

base_name = "scalers_pheno"
if restrict_24_hours:
    base_name += "-24"
else:
    base_name += "-all"

if include_notes:
    base_name += "-notes"

if include_cxr:
    base_name += "-cxr"

if include_ecg:
    base_name += "-ecg"
    
if include_missing:
    base_name += "-missingInd"

f_path = os.path.join(output_dir, f"{base_name}.pkl")
with open(f_path, 'wb') as f:
    pickle.dump(scalers, f)



import yaml

f_path = "hcup_ccs_2015_definitions.yaml"
with open(f_path, 'r') as f:
    hcup_ccs = yaml.safe_load(f)



f_path = "icd10cmtoicd9gem.csv"
icd10_to_icd9_df = pd.read_csv(f_path, low_memory=False)



benchmark_diags = {}

i = 0
for key in hcup_ccs.keys():
    curr_entry = hcup_ccs[key]

    if not curr_entry['use_in_benchmark']:
        continue
    
    curr_entry['icd9'] = curr_entry['codes']

    icd10_codes = []
    for code in curr_entry['codes']:
        curr_icd10_codes = icd10_to_icd9_df[icd10_to_icd9_df['icd9cm'] == code]['icd10cm'].values

        for icd10_code in curr_icd10_codes:
            icd10_codes.append(icd10_code)

    curr_entry['icd10'] = icd10_codes

    # Drop codes from curr_entry
    curr_entry.pop('codes')

    curr_entry['id'] = i
    benchmark_diags[key] = curr_entry
    i += 1


admissions_df = pd.read_csv(os.path.join(mimic_iv_path, "hosp", "admissions.csv"))
admissions_df = admissions_df.rename(columns={"hospital_expire_flag": "died"})
admissions_df = admissions_df[["subject_id", "hadm_id", "died"]]


d_icd_diagnoses = pd.read_csv(os.path.join(mimic_iv_path, "hosp", "d_icd_diagnoses.csv"))
diagnoses_df = pd.read_csv(os.path.join(mimic_iv_path, "hosp", "diagnoses_icd.csv"))

diagnoses_df = diagnoses_df.merge(d_icd_diagnoses, on=["icd_code", 'icd_version'], how="left")


def get_stay_list(stays):
    stays_list = []

    for curr_stay in tqdm(stays):
        curr_stay_ireg = ireg_vitals_ts_df[ireg_vitals_ts_df['stay_id'] == curr_stay].copy()
        curr_stay_imputed = imputed_vitals[imputed_vitals['stay_id'] == curr_stay].copy()

        if len(curr_stay_ireg) == 0:
            continue

        if include_notes:
            curr_stay_notes = notes_df[notes_df['stay_id'] == curr_stay].copy()

        if include_cxr:
            curr_stay_cxr = cxr_df[cxr_df['stay_id'] == curr_stay].copy()

        curr_stay_dict = {}
        curr_stay_dict['name'] = curr_stay_ireg['subject_id'].iloc[0]
        curr_stay_dict['hadm_id'] = curr_stay_ireg['hadm_id'].iloc[0]
        curr_stay_dict['stay_id'] = curr_stay
        curr_stay_dict['ts_tt'] = curr_stay_ireg['timedelta'].values

        curr_stay_ireg.drop(columns=['subject_id', 'hadm_id', 'stay_id', 'timedelta'], inplace=True)
        ireg_ts_mask = curr_stay_ireg.notnull()
        curr_stay_ireg.fillna(0, inplace=True)
        curr_stay_dict['irg_ts'] = curr_stay_ireg.values
        curr_stay_dict['irg_ts_mask'] = ireg_ts_mask.values.astype(int)

        curr_stay_imputed.drop(columns=['subject_id', 'hadm_id', 'stay_id', 'timedelta'], inplace=True)
        curr_stay_dict['reg_ts'] = curr_stay_imputed.values

        if include_notes:
            if len(curr_stay_notes) == 0:
                curr_stay_dict['text_data'] = []
                curr_stay_dict['text_time_to_end'] = []
                curr_stay_dict['text_missing'] = 1
                curr_stay_dict['text_embeddings'] = []
            else:
                curr_stay_dict['text_data'] = curr_stay_notes['text'].tolist()
                curr_stay_dict['text_time_to_end'] = curr_stay_notes['icu_time_delta'].values
                curr_stay_dict['text_embeddings'] = [emb[0][0] for emb in curr_stay_notes['biobert_embeddings']]
                curr_stay_dict['text_missing'] = 0
        
        if include_cxr:
            if len(curr_stay_cxr) == 0:
                curr_stay_dict['cxr_feats'] = []
                curr_stay_dict['cxr_time'] = []
                curr_stay_dict['cxr_missing'] = 1
            else:
                curr_stay_dict['cxr_feats'] = curr_stay_cxr['densefeatures'].tolist()
                curr_stay_dict['cxr_time'] = curr_stay_cxr['icu_time_delta'].values
                curr_stay_dict['cxr_missing'] = 0

        if include_ecg:
            curr_stay_ecg = ecg_df[ecg_df['stay_id'] == curr_stay].copy()
            if len(curr_stay_ecg) == 0:
                curr_stay_dict['ecg_feats'] = []
                curr_stay_dict['ecg_time'] = []
                curr_stay_dict['ecg_missing'] = 1
            else:
                curr_stay_dict['ecg_feats'] = curr_stay_ecg['embeddings'].tolist()
                curr_stay_dict['ecg_time'] = curr_stay_ecg['icu_time_delta'].values
                curr_stay_dict['ecg_missing'] = 0            

        curr_diagnoses = diagnoses_df[diagnoses_df['hadm_id'] == curr_stay_dict['hadm_id']]

        curr_labels = np.zeros(len(benchmark_diags.keys()))

        for index, row in curr_diagnoses.iterrows():
            for key in benchmark_diags.keys():
                curr_bench_diag = benchmark_diags[key]
                if (row['icd_version'] == 9) and (row['icd_code'] in curr_bench_diag['icd9']):
                    curr_labels[curr_bench_diag['id']] = 1
                elif (row['icd_version'] == 10) and (row['icd_code'] in curr_bench_diag['icd10']):
                    curr_labels[curr_bench_diag['id']] = 1

        curr_stay_dict['label'] = curr_labels

        stays_list.append(curr_stay_dict)

    return stays_list

train_stays_list = get_stay_list(train_stays)
val_stays_list = get_stay_list(val_stays)
test_stays_list = get_stay_list(test_stays)


# Save the data
import pickle

base_name = "pheno"

if restrict_24_hours:
    base_name += "-24"
else:
    base_name += "-all"

if include_notes:
    base_name += "-notes"

if include_cxr:
    base_name += "-cxr"

if include_ecg:
    base_name += "-ecg"
    
if include_missing:
    base_name += "-missingInd"

f_path = os.path.join(output_dir, f"train_{base_name}_stays.pkl")
with open(f_path, 'wb') as f:
    print(f"Saving train stays to {f_path}")
    pickle.dump(train_stays_list, f)

f_path = os.path.join(output_dir, f"val_{base_name}_stays.pkl")
with open(f_path, 'wb') as f:
    print(f"Saving val stays to {f_path}")
    pickle.dump(val_stays_list, f)

f_path = os.path.join(output_dir, f"test_{base_name}_stays.pkl")
with open(f_path, 'wb') as f:
    print(f"Saving test stays to {f_path}")
    pickle.dump(test_stays_list, f)


