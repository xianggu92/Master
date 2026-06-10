import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle  # 補上原本漏掉的 pickle 匯入

import time
import sys
import logging
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


def main():
    args = parse_args()

    accelerator = Accelerator(cpu=args.cpu)

    device = accelerator.device
    print('Using device:', device)
    os.makedirs(args.output_dir, exist_ok=True)

    make_save_dir(args)

    model = MULTCrossModel(args=args, device=device, orig_d_ts=30, orig_reg_d_ts=60, orig_d_txt=768, ts_seq_num=args.tt_max, text_seq_num=args.num_of_notes)
    with open(os.path.join(args.file_path, f'scalers_{args.task}.pkl'), 'rb') as f:
        scalers = pickle.load(f)

    model = accelerator.prepare(model)

    rootdir = args.ck_file_path
    seeds = [1] # list(range(1, 6))

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

            with torch.no_grad():
                # 1. 定義 100 個 Key 時間點 (0 到 1 均勻分佈)
                key_times = torch.linspace(0.0, 1.0, steps=100, device=device).unsqueeze(1) # Shape: (100, 1)
                times_np = key_times.squeeze(1).cpu().numpy()
                times_np = 1 - times_np

                # 時間特徵轉換函數 (Time2Vec)
                def get_time_embedding(t):
                    p_emb = model.periodic(t)
                    l_emb = model.linear(t)
                    return torch.cat([p_emb, l_emb], dim=-1)

                keys_emb = get_time_embedding(key_times) # Shape: (100, 64)
                
                # ----------------------------------------------------
                # 提取各模態的 Key 轉換矩陣 (來自 linears[1])
                # ----------------------------------------------------
                k_transformed_ts = model.time_attn_ts.linears[1](keys_emb) # Shape: (100, 64)
                d_k_ts = k_transformed_ts.size(-1)

                # 檢查模型是否包含文字模態，若有則提取文字的 Key 轉換
                has_text = hasattr(model, 'time_attn_text') and model.time_attn_text is not None
                if has_text:
                    k_transformed_text = model.time_attn_text.linears[1](keys_emb) # Shape: (100, 64)
                    d_k_text = k_transformed_text.size(-1)

                # 2. 定義三個想要測試的 Query 時間點
                queries = {
                    't = 0.0 (Start)': torch.tensor([[0.0]], dtype=torch.float32, device=device),
                    't = 0.5 (Mid)': torch.tensor([[0.5]], dtype=torch.float32, device=device),
                    't = 1.0 (End)': torch.tensor([[1.0]], dtype=torch.float32, device=device)
                }

                # 設定色彩與圖表基礎設定
                colors = {'t = 0.0 (Start)': '#E66101', 't = 0.5 (Mid)': '#5E3C99', 't = 1.0 (End)': '#0571B0'}
                os.makedirs(os.path.join(args.ck_file_path, 'images'), exist_ok=True)

                # ====================================================
                # 繪製 1. 時間序列 (Time Series) 的 Attention Weights
                # ====================================================
                plt.figure(figsize=(12, 6))
                for label, q_time in queries.items():
                    query_emb = get_time_embedding(q_time)
                    q_transformed = model.time_attn_ts.linears[0](query_emb)
                    
                    scores = torch.matmul(q_transformed, k_transformed_ts.transpose(0, 1)) / (d_k_ts ** 0.5)
                    attention_weights = F.softmax(scores, dim=-1).squeeze(0).cpu().numpy()
                    
                    plt.plot(times_np, attention_weights, label=f'Query: {label}', color=colors[label], linewidth=2)

                plt.title('MTAND Attention Weights Comparison for Time Series (t=0.0, 0.5, 1.0)', fontsize=14, pad=15)
                plt.xlabel('Key Time (0.0 to 1.0)', fontsize=12)
                plt.ylabel('Attention Score', fontsize=12)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.legend(loc='upper right', frameon=True)
                
                save_path_ts = os.path.join(args.ck_file_path, 'images', 'mtand_attention_ts.png')
                plt.savefig(save_path_ts, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"Successfully saved Time Series comparison plot to: {save_path_ts}")

                # ====================================================
                # 繪製 2. 文字模態 (Text) 的 Attention Weights (新增部分)
                # ====================================================
                if has_text:
                    plt.figure(figsize=(12, 6))
                    for label, q_time in queries.items():
                        query_emb = get_time_embedding(q_time)
                        # 使用文字模態專屬的 linears[0]
                        q_transformed = model.time_attn_text.linears[0](query_emb)
                        
                        # 與文字模態轉換後的 Key 計算點積注意力
                        scores = torch.matmul(q_transformed, k_transformed_text.transpose(0, 1)) / (d_k_text ** 0.5)
                        attention_weights = F.softmax(scores, dim=-1).squeeze(0).cpu().numpy()
                        
                        plt.plot(times_np, attention_weights, label=f'Query: {label}', color=colors[label], linewidth=2)

                    plt.title('MTAND Attention Weights Comparison for Text Modality (t=0.0, 0.5, 1.0)', fontsize=14, pad=15)
                    plt.xlabel('Key Time (0.0 to 1.0)', fontsize=12)
                    plt.ylabel('Attention Score', fontsize=12)
                    plt.grid(True, linestyle=':', alpha=0.6)
                    plt.legend(loc='upper right', frameon=True)
                    
                    save_path_text = os.path.join(args.ck_file_path, 'images', 'mtand_attention_text.png')
                    plt.savefig(save_path_text, bbox_inches='tight', dpi=300)
                    plt.close()
                    print(f"Successfully saved Text Modality comparison plot to: {save_path_text}")
                else:
                    print("Text modality (time_attn_text) is not enabled in this model config. Skipping text visualization.")


if __name__ == "__main__":
    main()