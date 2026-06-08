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


def main():
    args = parse_args()

    accelerator = Accelerator(cpu=args.cpu)

    device = accelerator.device
    print('Using device:', device)
    os.makedirs(args.output_dir, exist_ok=True)

    make_save_dir(args)
    os.makedirs(os.path.join(args.ck_file_path, 'log'), exist_ok=True)

    model = MULTCrossModel(args=args,device=device,orig_d_ts=30, orig_reg_d_ts=60, orig_d_txt=768, ts_seq_num=args.tt_max, text_seq_num=args.num_of_notes)
    with open(os.path.join(args.file_path, f'scalers_{args.task}.pkl'), 'rb') as f:
        scalers = pickle.load(f)

    model = \
    accelerator.prepare(model)

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

                # 時間特徵轉換函數 (periodic 63 + linear 1 = 64)
                def get_time_embedding(t):
                    p_emb = model.periodic(t)
                    l_emb = model.linear(t)
                    return torch.cat([p_emb, l_emb], dim=-1)

                keys_emb = get_time_embedding(key_times) # Shape: (100, 64)
                k_transformed = model.time_attn_ts.linears[1](keys_emb) # Shape: (100, 64)
                d_k = k_transformed.size(-1)

                # 2. 定義三個想要測試的 Query 時間點
                queries = {
                    't = 0.0 (Start)': torch.tensor([[0.0]], dtype=torch.float32, device=device),
                    't = 0.5 (Mid)': torch.tensor([[0.5]], dtype=torch.float32, device=device),
                    't = 1.0 (End)': torch.tensor([[1.0]], dtype=torch.float32, device=device)
                }

                # 3. 開始繪圖設定
                plt.figure(figsize=(12, 6))
                
                # 設定調色盤，讓三條線有明顯對比
                colors = {'t = 0.0 (Start)': '#E66101', 't = 0.5 (Mid)': '#5E3C99', 't = 1.0 (End)': '#0571B0'}
                texts = []

                # 4. 依序計算每個 Query 的注意力分數並畫線
                for label, q_time in queries.items():
                    query_emb = get_time_embedding(q_time) # Shape: (1, 64)
                    q_transformed = model.time_attn_ts.linears[0](query_emb) # Shape: (1, 64)
                    
                    # 計算點積注意力與 Softmax
                    scores = torch.matmul(q_transformed, k_transformed.transpose(0, 1)) / (d_k ** 0.5)
                    attention_weights = F.softmax(scores, dim=-1).squeeze(0).cpu().numpy()
                    
                    # 畫出該 Query 對應的注意力曲線
                    plt.plot(times_np, attention_weights, label=f'Query: {label}', color=colors[label], linewidth=2)

                # 5. 加上圖表美化與標籤
                plt.title('MTAND Attention Weights Comparison for Multiple Queries (t=0.0, 0.5, 1.0)', fontsize=14, pad=15)
                plt.xlabel('Key Time (0.0 to 1.0)', fontsize=12)
                plt.ylabel('Attention Score', fontsize=12)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.legend(loc='upper right', frameon=True)
                
                # 6. 自動調整文字位置避免重疊
                adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))
                
                # 7. 儲存圖表
                save_path = os.path.join(args.ck_file_path, 'images', 'mtand_attention.png')
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
                print(f"Successfully saved multi-query comparison plot to: {save_path}")




if __name__ == "__main__":
    main()