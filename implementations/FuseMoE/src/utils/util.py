import os
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../TS/mimic3-benchmarks')
sys.path.insert(0, '../ClinicalNotes_TimeSeries/models')
import pickle
import re
import numpy as np
import json
from preprocessing.data import *
import statistics as stat
logger = None
import  argparse
import pickle
from accelerate import Accelerator
from sklearn import metrics
import pdb
import logging
from datetime import datetime

from transformers import (AutoTokenizer,
                          AutoModel,
                          AutoConfig,
                          AdamW,
                          BertTokenizer,
                          BertModel,
                          get_scheduler,
                          set_seed,
                          BertPreTrainedModel,
                          LongformerConfig,
                          LongformerModel,
                          LongformerTokenizer,
                         )

def parse_args():
    parser = argparse.ArgumentParser(description="Alignment text and ts data", allow_abbrev=False)
    parser.add_argument(
            "--task", type=str, default="ihm"
        )
    parser.add_argument("--file_path", type=str, default="Data", help="A path to dataset folder")
    parser.add_argument("--output_dir", type=str, default="Checkpoints", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--mode", type=str, default="train", help="train/test")
    parser.add_argument("--modeltype", type=str, default="TS_Text", help="TS, Text or TS_Text")
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size  for the training dataloader.")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for the evaluation dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Total number of training epochs to perform.")
    parser.add_argument("--ts_learning_rate", type=float, default=0.0004, help="Initial learning rate for TS self-attention to use.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--kernel_size", type=int, default=1, help="Kernel size for CNN.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads.")
    parser.add_argument("--layers", type=int, default=3, help="Number of transformer encoder layer.")
    parser.add_argument("--embed_dim", default=30, type=int, help="attention embedding dim.")
    parser.add_argument("--irregular_learn_emb_ts", type=str, default=None)
    parser.add_argument("--irregular_learn_emb_text", type=str, default=None)
    parser.add_argument("--irregular_learn_emb_cxr", type=str, default=None)
    parser.add_argument("--irregular_learn_emb_ecg", type=str, default=None)
    parser.add_argument("--reg_ts", action='store_true')
    parser.add_argument("--tt_max", default=48, type=int, help="max time for irregular time series.")
    parser.add_argument("--orig_d_ts", default=30, type=int, help="Number of time series variables.")
    parser.add_argument("--orig_d_txt", default=768, type=int, help="Dimention of text embeddings.")
    parser.add_argument("--embed_time", default=64, type=int, help="emdedding for time.")
    parser.add_argument("--dropout", default=0.10, type=float, help="dropout.")
    parser.add_argument("--n_patch", default=2, type=int, help="Number of patches in patch interpolation.")
    parser.add_argument("--n_ref_point", default=48, type=int, help="Number of reference points in patch interpolation.")
    parser.add_argument("--use_global", action='store_true', help="Use global interpolation in patch interpolation.")

    parser.add_argument('--TS_mixup', action='store_true', help='mix up reg and irg data')
    parser.add_argument("--mixup_level", default=None, type=str, help="mixedup level for two time series data, choose: 'batch', batch_seq' or 'batch_seq_feature'. ")

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--debug", action='store_true')

    parser.add_argument("--cross_method", default='moe', type=str, help="all fusion methods: moe, hme, moe_cross, self_cross, MAGGate, MulT, Outer, concat")
    parser.add_argument("--hidden_size", default=512, type=int, help="hidden size of MLP second layer")
    parser.add_argument("--gating_function", nargs='*', type=str, help="all gating functions: softmax, laplace, gaussian, enter at least one")
    parser.add_argument("--num_of_experts", nargs='*', type=int, help="number of MLPs in MoE, for HME need to specify each level")
    parser.add_argument("--top_k", nargs='*', type=int, help="the number of experts finally combined together for joint and permod routers")
    parser.add_argument("--disjoint_top_k", default=2, type=int, help="the number of experts finally combined together for disjoint routers")
    parser.add_argument("--num_modalities", default=2, type=int, help="the number of input modalities used to train transformer")
    parser.add_argument("--router_type", default='joint', type=str, help="all router types: joint, permod, disjoint")
    parser.add_argument("--use_balance_loss", action='store_true', help="Whether to include balance_loss term in total loss (only for MoE/HME fusion methods)")
    parser.add_argument("--balance_loss_coef", default=0.01, type=float, help="Coefficient for balance_loss term in total loss")
    args = parser.parse_known_args()[0]
    return args

def loadBert(args,device):
    if args.model_name!=None:
        if args.model_name== 'BioBert':
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            BioBert=AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif args.model_name=="bioRoberta":
            config = AutoConfig.from_pretrained("allenai/biomed_roberta_base", num_labels=args.num_labels)
            tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")
            BioBert = AutoModel.from_pretrained("allenai/biomed_roberta_base")
        elif args.model_name== "Bert":
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            BioBert = BertModel.from_pretrained("bert-base-uncased")
        elif args.model_name== "bioLongformer":
            tokenizer = AutoTokenizer.from_pretrained("/mnt/data/yihua/.cache/huggingface/hub/models--yikuan8--Clinical-Longformer/snapshots/5e2ebe8f6e98d4751eaf71a8514e0edbe73989a2")
            # BioBert= AutoModel.from_pretrained("/mnt/data/yihua/.cache/huggingface/hub/models--yikuan8--Clinical-Longformer/snapshots/5e2ebe8f6e98d4751eaf71a8514e0edbe73989a2")

        else:
            raise ValueError("model_name should be BioBert,bioRoberta,bioLongformer or Bert")
    else:
        if args.model_path!=None:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            BioBert = AutoModel.from_pretrained(args.model_path)
        else:
            raise ValueError("provide either model_name or model_path")

    BioBert = None # BioBert.to(device)
    BioBertConfig = None # BioBert.config
    return BioBert, BioBertConfig, tokenizer


def data_generate(args):
    dataPath = os.path.join(args.file_path,  'all_data_p2x_data.pkl')
    if os.path.isfile(dataPath):
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            data = pickle.load(f)
            if args.debug:
                data=data[:100]

    data=np.array(data)
    total_num=len(data)
    idx=np.arange(total_num)

    np.random.seed(args.seed)
    np.random.shuffle(idx)

    train= data[idx[:int(len(idx)*0.8)]]
    print(train[0]['data_names'])
    val=data[idx[int(len(idx)*0.8):int(len(idx)*0.9)]]
    test=data[idx[int(len(idx)*0.9):]]

    train=train.tolist()
    val=val.tolist()
    test=test.tolist()
    return train, val, test


def metrics_multilabel(y_true, predictions, verbose=1):
    # import pdb; pdb.set_trace()
    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
                                          average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
                                          average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
                                             average="weighted")

    if verbose:
        # print("ROC AUC scores for labels:", auc_scores)
        print("ave_auc_micro = {}".format(ave_auc_micro))
        print("ave_auc_macro = {}".format(ave_auc_macro))
        print("ave_auc_weighted = {}".format(ave_auc_weighted))

    return{"auc_scores": auc_scores,
            "ave_auc_micro": ave_auc_micro,
            "ave_auc_macro": ave_auc_macro,
            "ave_auc_weighted": ave_auc_weighted}


def diff_float(time1, time2):
    h = (time2-time1).astype('timedelta64[m]').astype(int)
    return h/60.0


def get_time_to_end_diffs(times, starttimes):

    timetoends = []
    for times, st in zip(times, starttimes):
        difftimes = []
        et = np.datetime64(st) + np.timedelta64(49, 'h')
        for t in times:
            time = np.datetime64(t)
            dt = diff_float(time, et)
            assert dt >= 0 #delta t should be positive
            difftimes.append(dt)
        timetoends.append(difftimes)
    return timetoends

def change_data_form(file_path,mode,debug=False):
    dataPath = os.path.join(file_path, mode + '.pkl')
    if os.path.isfile(dataPath):
        # We write the processed data to a pkl file so if we did that already we do not have to pre-process again and this increases the running speed significantly
        print('Using', dataPath)
        with open(dataPath, 'rb') as f:
            # (data, _, _, _) = pickle.load(f)
            data = pickle.load(f)
            if debug:
                data=data[:500]

        data_X = data[0]
        data_y = data[1]
        data_text = data[2]
        data_names = data[3]
        start_times = data[4]
        timetoends = data[5]

        dataList=[]

        assert len(data_X)==len(data_y)==len(data_text)==len(data_names)==len(start_times)==len(timetoends) 


        assert  len(data_text[0])==len(timetoends[0])
        for x,y, text, name, start, end in zip(data_X,data_y,data_text, data_names,start_times,timetoends):
            if len(text)==0:
                continue
            new_text=[]
            for t in text:
                # import pdb;
                # pdb.set_trace()
                t=re.sub(r'\s([,;?.!:%"](?:\s|$))', r'\1', t)
                t=re.sub(r"\b\s+'\b", r"'", t)
                new_text.append(t.lower().strip())


            data_detail={"data_names":name,
                         "TS_data":x,
                         "text_data":new_text,
                        "label":y,
                         "adm_time":start,
                         "text_time_to_end":end
                        }
            dataList.append(data_detail)

    os.makedirs('Data',exist_ok=True)
    dataPath2 = os.path.join(file_path, mode + 'p2x_data.pkl')

    with open(dataPath2, 'wb') as f:
        # Write the processed data to pickle file so it is faster to just read later
        pickle.dump(dataList, f)

    return dataList

def data_replace(file_path1,file_path2,mode,debug=False):
    dataPath1 = os.path.join(file_path2, mode + '.pkl')
    dataPath2 = os.path.join(file_path1, mode + 'p2x_data.pkl')
    if os.path.isfile(dataPath1):
        # We write the processed data to a pkl file so if we did that already we do not have to pre-process again and this increases the running speed significantly
        print('Using', dataPath1)
        with open(dataPath1, 'rb') as f:
            data = pickle.load(f)
            if debug:
                data=data[:500]

    with open(dataPath2, 'rb') as f:
            data_r=pickle.load(f)
    data_X = data[0]
    data_y = data[1]
    data_text = data[2]
    data_names = data[3]
    start_times = data[4]
    timetoends = data[5]
    data_dict={}

    assert len(data_X)==len(data_y)==len(data_text)==len(data_names)==len(start_times)==len(timetoends) 
    assert  len(data_text[0])==len(timetoends[0])
    for x,name in zip(data_X, data_names):

        data_dict[name]=x
    for idx, data_detail in enumerate(data_r):
        new_x=data_dict[data_detail['data_names']]
        data_detail['TS_data']=new_x

    dataPath3=os.path.join(file_path2, mode + 'p2x_data.pkl')
    with open(dataPath3, 'wb') as f:
        pickle.dump(data_r, f)


def merge_reg_irg(dataPath_reg, dataPath_irg):
    with open(dataPath_irg, 'rb') as f:
        data_irg=pickle.load(f)

    with open(dataPath_reg, 'rb') as f:
        data_reg=pickle.load(f)

    for idx, data_dict in enumerate(data_reg):
        irg_dict=data_irg[data_dict['data_names']]
        data_dict['ts_tt']=irg_dict['ts_tt']
        data_dict['irg_ts']=irg_dict['irg_ts']
        data_dict['irg_ts_mask']=irg_dict['irg_ts_mask']

        assert (data_dict['label']==irg_dict['label']).all()

    with open(dataPath_reg, 'wb') as f:
        pickle.dump(data_reg,f)


def get_logger(args):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(os.path.join(args.ck_file_path, 'log', str(args.seed) +'.log'), mode="w")
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger