# %%
import os
import pandas as pd
# from MIMIC_IV_HAIM_API import *
import torchxrayvision as xrv
import skimage
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

# %%
mimic_iv_cxr_parent = "/mnt/nfs_share/Public_Data/Dataset_MIMIC_CXR_JPG/physionet.org/files/mimic-cxr-jpg/2.0.0"
mm_dir = "/mnt/data/yihua/master/datasets/mimic-iv"
device = 'cuda:2'

output_dir = os.path.join(mm_dir, "preprocessing")
os.makedirs(output_dir, exist_ok=True)

# %%
f_path = os.path.join(mimic_iv_cxr_parent, "mimic-cxr-2.0.0-metadata.csv")
meta_data_df = pd.read_csv(f_path, low_memory=False)

# %%
# meta_data_df = meta_data_df[:100]

# %% [markdown]
# Extract densefeatures and predictions

# %%
meta_data_df['densefeatures'] = None
meta_data_df['predictions'] = None

model_weights_name = "densenet121-res224-chex" 
model = xrv.models.DenseNet(weights = model_weights_name).to(device)

for index, row in tqdm(meta_data_df.iterrows(), total=meta_data_df.shape[0]):
    curr_subject_id = int(row['subject_id'])
    curr_study_id = int(row['study_id'])
    curr_dicom_id = row['dicom_id']

    f_subfolder = "p" + str(curr_subject_id)[0:2]
    pt_folder = "p" + str(curr_subject_id)
    s_folder = "s" + str(curr_study_id)
    curr_f_path = os.path.join(mimic_iv_cxr_parent, 'files', f_subfolder, pt_folder, s_folder, curr_dicom_id + ".jpg")

    if os.path.exists(curr_f_path):
        img = skimage.io.imread(curr_f_path)

        img = xrv.datasets.normalize(img, 255)
        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)   
        img = img[None, :, :]
        
        with torch.no_grad():
            img = torch.from_numpy(img).unsqueeze(0)
            # if cuda:
            img = img.to(device)
            
            # Extract dense features
            feats = model.features(img)
            feats = F.relu(feats, inplace=True)
            feats = F.adaptive_avg_pool2d(feats, (1, 1))
            densefeatures = feats.cpu().detach().numpy().reshape(-1)
            meta_data_df.at[index, 'densefeatures'] = densefeatures # append to list of dense features for all images

            preds = model(img).cpu()
            predictions = preds[0].detach().numpy()
            meta_data_df.at[index, 'predictions'] = predictions

# %%
cols_to_drop = ['Unnamed: 0', 'Note_folder', 'Note_file', 'Note', 'Img_Folder',\
     'Img_Filename', 'Rows', 'Columns', 'StudyDate', 'StudyTime', 'StudyDateForm', \
        'StudyTimeForm']

for col in cols_to_drop:
    if col in meta_data_df.columns:
        meta_data_df.drop(columns=[col], inplace=True)

# %%
f_path = os.path.join(output_dir, "cxr_embeddings.pkl")
meta_data_df.to_pickle(f_path)
print(f_path)

# %%
f_path = os.path.join(output_dir, "cxr_embeddings.pkl")
df = pd.read_pickle(f_path)


