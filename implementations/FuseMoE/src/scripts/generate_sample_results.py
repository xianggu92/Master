import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import sys
import logging
import os
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
    os.makedirs(os.path.join(args.ck_file_path, 'log'), exist_ok=True)

    model = MULTCrossModel(args=args,device=device,orig_d_ts=30, orig_reg_d_ts=60, orig_d_txt=768, ts_seq_num=args.tt_max, text_seq_num=args.num_of_notes)
    with open(os.path.join(args.file_path, f'scalers_{args.task}.pkl'), 'rb') as f:
        scalers = pickle.load(f)

    features = []
    def hook(module, input, output):
        features.append(input[0].cpu())

    model.proj1.register_forward_hook(hook)

    model, test_data_loader = \
    accelerator.prepare(model, test_data_loader)

    rootdir = args.ck_file_path
    seeds = [0] 
    # seeds = list(range(1, 6))
    all_pred_list = []
    all_label_list = []
    cnt = defaultdict(int)

    variable_names = ['Anion Gap',
        'Bicarbonate', 'Calcium, Total', 'Chloride', 'Creatinine',
        'Diastolic BP', 'GCS - Eye Opening', 'GCS - Motor Response',
        'GCS - Verbal Response', 'Glucose', 'Heart Rate', 'Hematocrit',
        'Hemoglobin', 'MCH', 'MCHC', 'MCV', 'Magnesium', 'Mean BP',
        'Neutrophils', 'O2 Saturation', 'Phosphate', 'Platelet Count', 'RDW',
        'Red Blood Cells', 'Respiratory Rate', 'Sodium', 'Systolic BP',
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
                labels = batch.pop('labels')
                with torch.no_grad():
                    logits = model(**batch)

                    if logits.dim() == 1:
                        logits = logits.unsqueeze(-1) # [B, 1]

                    all_logits.append(logits.cpu().numpy())
                    all_label.append(labels.cpu().numpy())

                # === 樣本視覺化 ===

                # 計算預測標籤
                if 'ihm' in args.task:
                    pred_prob = logits[0].item()
                    pred_cls = 1 if pred_prob > 0.5 else 0
                    true_cls = labels[0].item()

                    if seed == 1 and cnt[(true_cls, pred_cls)] < 5:
                        cnt[(true_cls, pred_cls)] += 1

                        # # 取出當前 batch 的第一個樣本 (index 0) 進行視覺化
                        # raw_ts = batch['x_ts'][0].cpu().numpy()
                        # raw_time = args.tt_max - batch['ts_tt_list'][0].cpu().numpy() * args.tt_max
                        # raw_mask = batch['x_ts_mask'][0].cpu().numpy()
                        
                        # # 提取文字（Notes）的時間與實際文本列表
                        # note_time_ratio = batch['note_time_list'][0].cpu().numpy()
                        # note_mask = batch['note_time_mask_list'][0].cpu().numpy()
                        # actual_texts = batch['text_data'][0] 

                        # # 篩選出真正有文字紀錄的時間點 (mask == 1)
                        # valid_note_idx = np.where(note_mask == 1)[0]
                        # actual_note_times = args.tt_max - (1 - note_time_ratio[valid_note_idx]) * args.tt_max
                        
                        # is_correct = "Correct" if pred_cls == true_cls else "Incorrect"
                        # title_text = f"Sample {idx} - {args.task} Prediction: {is_correct} (Pred: {pred_prob:.3f}, True: {true_cls})"
                        
                        # # 開始畫圖
                        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), sharex=True, 
                        #                             gridspec_kw={'height_ratios': [1, 1]})
                        
                        # # --- 子圖 1：不規則生理訊號折線 ---
                        # num_features_to_plot = raw_ts.shape[1]
                        # cmap = plt.cm.get_cmap()
                        # texts = []

                        # for f_idx in range(num_features_to_plot):
                        #     valid_ts_idx = np.where(raw_mask[:, f_idx] == 1)[0]
                        #     if len(valid_ts_idx) > 0:
                        #         t_points = raw_time[valid_ts_idx]
                        #         v_points = scalers[variable_names[f_idx]].inverse_transform(raw_ts[valid_ts_idx, f_idx].reshape(-1, 1))
                        #         color = cmap(f_idx / num_features_to_plot)

                        #         ax1.plot(t_points, v_points, 
                        #                 marker='o', linestyle='-', label=f'{variable_names[f_idx]}', color=color)
                                
                        #         last_t = t_points[0]
                        #         last_v = v_points[0]

                        #         t_obj = ax1.text(last_t + 0.3, last_v, f'{variable_names[f_idx]}', 
                        #                 fontsize=8, weight='bold', color=color,
                        #                 va='center', ha='left')
                        #         texts.append(t_obj)

                        # adjust_text(texts, ax=ax1, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
                        # ax1.set_title(title_text, fontsize=14, color='green' if is_correct == "Correct" else 'red', pad=15)
                        # ax1.set_ylabel("Vital Sign Values")
                        # ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1), ncol=2, 
                        #         borderaxespad=0, fontsize=9, frameon=True)
                        # ax1.grid(True, linestyle='--', alpha=0.5)
                        
                        # # --- 子圖 2：臨床筆記文字時間線 ---
                        # # 畫一條基準水平線代表時間軸
                        # ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
                        
                        # # 交錯上下擺放標籤 (給予更大的 offset 避免蓋到生理訊號圖)
                        # plot_effects = [0.5, -0.5] 
                        
                        # for i, t_note in enumerate(actual_note_times):
                        #     if i >= len(actual_texts): # 安全防護，避免 text 數量不對齊
                        #         break
                                
                        #     offset = plot_effects[i % 2]
                        #     # 畫垂直引線
                        #     ax2.vlines(x=t_note, ymin=0, ymax=offset, color='darkorange', linestyle='--', alpha=0.7)
                        #     # 畫時間線上的錨點
                        #     ax2.plot(t_note, 0, marker='s', color='darkorange', mfc='white', ms=8, mew=2)
                            
                        #     # 處理文本：截斷過長文字並自動折行
                        #     full_text = actual_texts[i]
                        #     text_max_length = 150
                        #     truncated_text = full_text if len(full_text) <= text_max_length else full_text[:text_max_length] + "..."
                        #     wrapped_text = textwrap.fill(truncated_text, width=20) # 自動換行
                            
                        #     box_content = f"t={t_note:.1f}\n{wrapped_text}"
                            
                        #     # 標註文字氣泡框
                        #     ax2.text(t_note, offset + (0.08 if offset > 0 else -0.08), box_content, 
                        #             ha='center', va='center', fontsize=9.5,
                        #             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0', alpha=0.9, edgecolor='darkorange'))
                        
                        # ax2.set_ylim(-1.2, 1.2)
                        # ax2.get_yaxis().set_visible(False) # 隱藏 Y 軸
                        # ax2.set_xlabel("Timeline (Hours / Timesteps)")
                        # ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
                        
                        # os.makedirs(os.path.join(args.ck_file_path, 'images'), exist_ok=True)
                        # plt.savefig(os.path.join(args.ck_file_path, 'images', f"ts_real_text_timeline_{pred_cls}_{true_cls}_{cnt[(true_cls, pred_cls)]-1}.png"), bbox_inches='tight')
                        # plt.show()
                        # plt.close()

            all_logits = np.concatenate(all_logits, axis=0)
            all_label = np.concatenate(all_label, axis=0)
            all_pred = np.where(all_logits > 0.5, 1, 0)
            all_pred_list.append(all_pred)
            all_label_list.append(all_label)


    features = torch.cat(features, dim=0).numpy()
    all_pred_list = np.concatenate(all_pred_list, axis=0)
    all_label_list = np.concatenate(all_label_list, axis=0)

    if len(seeds) == 5:
        with open(rootdir + "/sample_result.pkl","wb") as f:
            result = {'features': features, 'preds': all_pred_list, 'labels': all_label_list}
            pickle.dump(result, f)


if __name__ == "__main__":
    main()