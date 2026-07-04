import re
import os
import torch
import operator
from statistics import mean,stdev
import fnmatch

import shutil


def save_checkpoint(state, is_best, filename):
    """Save checkpoint if a new best is achieved"""
    if is_best:
#         print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")

def make_save_dir(args):

    output_dir=args.output_dir + "/" + args.task + "_" + args.modeltype

    if args.irregular_learn_emb_ts is not None and "TS" in args.modeltype:
        output_dir += "_TS_" + args.irregular_learn_emb_ts + "_" + str(args.embed_time)
    if args.irregular_learn_emb_text is not None and 'Text' in args.modeltype:
        output_dir += "_Text_" + args.irregular_learn_emb_text + "_" + str(args.embed_time)
    if args.irregular_learn_emb_cxr is not None and "CXR" in args.modeltype:
        output_dir += "_CXR_" + args.irregular_learn_emb_cxr + "_" + str(args.embed_time)
    if args.irregular_learn_emb_ecg is not None and 'ECG' in args.modeltype:
        output_dir += "_ECG_" + args.irregular_learn_emb_ecg + "_" + str(args.embed_time)

    if args.irregular_learn_emb_ts == 'PatchInterpolation':
        output_dir += '_' + str(args.n_patch) + '_' + str(args.n_ref_point)

        if args.use_global:
            output_dir += '_global'

    if args.num_modalities > 1:
        output_dir += '_layer' + str(args.layers)
        output_dir+= "_" + args.cross_method

        if args.cross_method == 'moe':
            output_dir += f"_{args.gating_function}"
            output_dir += f"_{args.router_type}"
            output_dir += f"_{args.num_of_experts}"
            output_dir += f"_top_{args.top_k}"
            if args.router_type == 'disjoint':
                output_dir += f"_disjoint_{args.disjoint_top_k}"

    if args.TS_mixup:
        output_dir += "_" + args.mixup_level

    output_dir += "_" + str(args.ts_learning_rate) + "_" + str(args.num_train_epochs) + "_" + str(args.num_heads) + "_" + str(args.embed_dim) +\
        "_" + str(args.kernel_size) + "_" + str(args.train_batch_size) + "_" + str(args.hidden_size) + '/'

    args.ck_file_path = output_dir
    os.makedirs(output_dir,  exist_ok=True)


def check_point(all_val, model, all_logits, args, eval_score=None):
    output_dir = args.ck_file_path

    seed = args.seed

    if eval_score:
        output_dir += eval_score +'/'
    os.makedirs(output_dir, exist_ok=True)

    filename = output_dir+str(seed)+'.pth.tar'

    if not os.path.exists(filename):
        is_best = True
        save_checkpoint({
        'network':model.state_dict(),
        'logits':all_logits,
        'best_val': all_val,
        'args': args}, is_best, filename)
    else:
        checkpoint = torch.load(filename)
        # import pdb; pdb.set_trace()
        val = checkpoint['best_val'][eval_score]
        best_val= all_val[eval_score]
        is_best = bool(best_val>val)
        if is_best:
            save_checkpoint({
            'network':model.state_dict(),
            'logits':all_logits,
            'best_val': all_val,
            'args': args}, is_best, filename)