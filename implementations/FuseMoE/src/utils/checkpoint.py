import json
import os
import torch


def save_checkpoint(state, is_best, filename):
    """Save checkpoint if a new best is achieved"""
    if is_best:
#         print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")

def make_save_dir(args):
    output_dir = os.path.join(
        args.output_dir,
        args.modeltype,
        args.task,
        args.run_name,
    )

    # Keep a trailing separator because checkpoint callers append metric names
    # and filenames to ck_file_path.
    args.ck_file_path = output_dir + os.sep
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(output_dir, "config.json")
    config = vars(args).copy()
    config.pop("seed", None)
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, ensure_ascii=False, default=str)


def check_point(all_val, model, all_logits, args, eval_score=None):
    output_dir = args.ck_file_path

    seed = args.seed

    os.makedirs(output_dir, exist_ok=True)

    filename = output_dir+str(seed)+'.pth.tar'
    save_checkpoint({
        'network': model.state_dict(),
        'logits': all_logits,
        'best_val': all_val,
        'args': args,
    }, True, filename)
