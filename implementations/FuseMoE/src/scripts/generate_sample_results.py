import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import sys
import logging
import os
from tqdm import tqdm
import pickle
import numpy as np

logger = logging.getLogger(__name__)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.model import *
from core.train import *
from utils.checkpoint import *
from utils.util import *
from accelerate import Accelerator
from core.interp import *
from preprocessing.data_mimiciv import data_perpare
import matplotlib.pyplot as plt
from adjustText import adjust_text
import textwrap
from collections import defaultdict

def main():
    args = parse_args()

    accelerator = Accelerator(cpu=args.cpu)
    device = accelerator.device
    print('Using device:', device)
    os.makedirs(args.output_dir, exist_ok=True)

    _, _, test_data_loader = data_perpare(args, 'test')

    make_save_dir(args)

    model = MULTCrossModel(args=args,device=device,orig_d_ts=30, orig_reg_d_ts=60, orig_d_txt=768, ts_seq_num=args.tt_max, text_seq_num=args.num_of_notes)
    with open(os.path.join(args.file_path, f'scalers_{args.task}.pkl'), 'rb') as f:
        scalers = pickle.load(f)

    # Hook 分類特徵
    features = []
    def hook(module, input, output):
        features.append(input[0].cpu())

    model.proj1.register_forward_hook(hook)

    # Hook 時間序列插補結果
    ts_mtand_outputs = []
    def ts_mtand_hook(module, input, output):
        # output shape: [B, tt_max, h * dim]
        ts_mtand_outputs.append(output.cpu())

    if hasattr(model, 'time_attn_ts'):
        model.time_attn_ts.head_output_identity.register_forward_hook(ts_mtand_hook)

    # Hook 文字插補結果
    text_mtand_outputs = []
    def text_mtand_hook(module, input, output):
        # output shape: [B, tt_max, h * dim]
        text_mtand_outputs.append(output.cpu())

    if hasattr(model, 'time_attn_text'):
        model.time_attn_text.head_output_identity.register_forward_hook(text_mtand_hook)

    model, test_data_loader = \
    accelerator.prepare(model, test_data_loader)

    rootdir = args.ck_file_path
    seeds = [1] 
    all_pred_list = []
    all_label_list = []
    sample_count = defaultdict(int)

    variable_names = ['Anion Gap', 'Bicarbonate', 'Calcium, Total', 'Chloride', 'Creatinine',
                      'Diastolic BP', 'GCS - Eye Opening', 'GCS - Motor Response', 'GCS - Verbal Response',
                      'Glucose', 'Heart Rate', 'Hematocrit', 'Hemoglobin', 'MCH', 'MCHC', 'MCV',
                      'Magnesium', 'Mean BP', 'Neutrophils', 'O2 Saturation', 'Phosphate', 'Platelet Count',
                      'RDW', 'Red Blood Cells', 'Respiratory Rate', 'Sodium', 'Systolic BP',
                      'Urea Nitrogen', 'Vancomycin', 'White Blood Cells']

    for seed in seeds:
        for subdir, dirs, files in os.walk(rootdir):
            substr = subdir.split('/')[-1]
            if 'f1' not in substr:
                continue

            file = f'{seed}.pth.tar'
            file_path = os.path.join(subdir, file)
            print(file_path)
            checkpoint = torch.load(file_path, map_location=device)
            model.load_state_dict(checkpoint['network'])
            model.eval()

            all_logits = []
            all_label = []
            
            for idx, batch in enumerate(tqdm(test_data_loader)):
                # 清空上一次 batch 的暫存
                ts_mtand_outputs.clear() 
                
                # 複製一份真實的輸入序列用來畫圖對比
                raw_x_ts = batch['x_ts'].cpu().numpy()
                raw_ts_tt = batch['ts_tt_list'].cpu().numpy()
                raw_text_emb = batch.get('text_emb', None)
                raw_note_time = batch.get('note_time_list', None)
                raw_note_mask = batch.get('note_time_mask_list', None)
                if raw_text_emb is not None:
                    raw_text_emb = raw_text_emb.cpu().numpy()
                if raw_note_time is not None:
                    raw_note_time = raw_note_time.cpu().numpy()
                if raw_note_mask is not None:
                    raw_note_mask = raw_note_mask.cpu().numpy()
                
                labels = batch.pop('labels')
                with torch.no_grad():
                    logits = model(**batch)

                    if logits.dim() == 1:
                        logits = logits.unsqueeze(-1) # [B, 1]

                    all_logits.append(logits.cpu().numpy())
                    all_label.append(labels.cpu().numpy())
                
                pred_prob = logits[0].item()
                pred_cls = 1 if pred_prob > 0.5 else 0
                true_cls = labels[0].item()

                if sample_count[(true_cls, pred_cls)] < 5:
                    sample_count[(true_cls, pred_cls)] += 1

                    # mTAND 填補後的時間點 (0 到 1 的均勻分佈)
                    query_time = np.linspace(0, 1, args.tt_max)
                    suffix = f'T{true_cls}_P{pred_cls}_cnt{sample_count[(true_cls, pred_cls)]}.png'

                    # =====================================
                    # 時間序列模態的插補視覺化
                    # =====================================

                    dim = model.orig_d_ts * 2
                    current_mtand_res = ts_mtand_outputs[0][0, :, :dim].numpy()
                    
                    # 前半段為填補後的數值 (Value)，後半段為填補後的 Mask
                    imputed_values = current_mtand_res[:, :model.orig_d_ts]
                    
                    # 建立畫布
                    selected_features_idx = range(30)
                    
                    fig, axes = plt.subplots(len(selected_features_idx), 1, figsize=(12, 3 * len(selected_features_idx)), sharex=True)
                    if len(selected_features_idx) == 1:
                        axes = [axes]
                        
                    # 原始不規則採樣時間點 (也是 0 到 1)
                    raw_time = raw_ts_tt[0] 

                    for i, feat_idx in enumerate(selected_features_idx):
                        feat_name = variable_names[feat_idx]
                        ax = axes[i]
                        
                        # 繪製真實不規則採樣的點
                        ax.scatter(raw_time, raw_x_ts[0, :, feat_idx], color='red', label='Observed Points', zorder=5, s=40)
                        
                        # 繪製 mTAND 第一個頭的插值的點
                        ax.scatter(query_time, imputed_values[:, feat_idx], color='blue', linestyle='-', label='mTAND Head-1 Imputation', alpha=0.8)
                        
                        ax.set_ylabel(feat_name)
                        ax.grid(True, linestyle='--', alpha=0.5)
                        if i == 0:
                            ax.legend(loc='upper left')
                    
                    axes[-1].set_xlabel('Normalized Time (0 to 1)')
                    plt.suptitle(f'Time Series Imputation Analaysis (True Cls: {true_cls}, Pred Cls: {pred_cls}, Prob: {pred_prob:.4f})', fontsize=14)
                    plt.tight_layout(rect=[0, 0, 1, 0.98]) # 留出頂部給標題
                    
                    # 儲存圖片
                    plot_save_path = os.path.join(rootdir, 'images', f'ts_imputation_{suffix}')
                    plt.savefig(plot_save_path, dpi=150)
                    plt.close()

                    # =========================================================
                    # 文字模態的插補視覺化
                    # =========================================================

                    if len(text_mtand_outputs) > 0 and raw_note_time is not None:
                        # 文字模態原始的特徵維度固定為 768
                        orig_d_txt = 768
                        
                        # 每個頭的特徵維度（與時間序列相同，前 orig_d_txt 個元素代表第一個頭的 Value）
                        imputed_text_values = text_mtand_outputs[0][0, :, :orig_d_txt].numpy()
                        
                        # 選擇前 N 個隱含特徵維度進行視覺化（例如前 8 維）
                        num_plot_features = min(8, orig_d_txt)
                        
                        fig, axes = plt.subplots(num_plot_features, 1, figsize=(12, 2.5 * num_plot_features), sharex=True)
                        if num_plot_features == 1:
                            axes = [axes]
                        
                        # 獲取該樣本真實有紀錄臨床文字的時間點與原始 Embedding
                        valid_note_mask = raw_note_mask[0] == 1 if raw_note_mask is not None else np.ones_like(raw_note_time[0], dtype=bool)
                        observed_note_times = raw_note_time[0][valid_note_mask]
                        observed_text_emb = raw_text_emb[0][valid_note_mask] # 取出有效時間點的原始文字特徵
                        
                        for i in range(num_plot_features):
                            ax = axes[i]
                            
                            # 繪製真實觀測到的文字 Embedding 數值
                            ax.scatter(observed_note_times, observed_text_emb[:, i], color='red', label='Observed Notes', zorder=5, s=40)
                            
                            # 繪製 mTAND 第一個頭對該文字隱含特徵插補出的數值
                            ax.scatter(query_time, imputed_text_values[:, i], color='purple', linestyle='-', label='mTAND Head-1 Text Imputation', alpha=0.8)
                            
                            ax.set_ylabel(f'Latent Dim {i}', fontsize=9)
                            ax.grid(True, linestyle='--', alpha=0.5)
                            if i == 0:
                                ax.legend(loc='upper left')
                        
                        axes[-1].set_xlabel('Normalized Time (0 to 1)')
                        plt.suptitle(f'Text Modality Imputation Analysis (True Cls: {true_cls}, Pred Cls: {pred_cls}, Prob: {pred_prob:.4f})', fontsize=14)
                        plt.tight_layout(rect=[0, 0, 1, 0.98])
                        
                        text_save_path = os.path.join(rootdir, 'images', f'text_imputation_{suffix}')
                        plt.savefig(text_save_path, dpi=150)
                        plt.close()

            all_logits = np.concatenate(all_logits, axis=0)
            all_label = np.concatenate(all_label, axis=0)
            all_pred = np.where(all_logits > 0.5, 1, 0)
            all_pred_list.append(all_pred)
            all_label_list.append(all_label)

    features = torch.cat(features, dim=0).numpy()
    all_pred_list = np.concatenate(all_pred_list, axis=0)
    all_label_list = np.concatenate(all_label_list, axis=0)

    # 測試五個種子時才儲存樣本結果
    if len(seeds) == 5:
        with open(rootdir + "/sample_result.pkl","wb") as f:
            result = {'features': features, 'preds': all_pred_list, 'labels': all_label_list}
            pickle.dump(result, f)


if __name__ == "__main__":
    main()