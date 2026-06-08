import os
import torch

import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

from core.model import *
from core.train import *
from utils.checkpoint import *
from utils.util import *
from accelerate import Accelerator
from preprocessing.data_mimiciv import data_perpare
import wandb



def main():
    args = parse_args()

    if args.fp16:
        args.mixed_precision="fp16"
    else:
        args.mixed_precision="no"

    accelerator = Accelerator(mixed_precision=args.mixed_precision,cpu=args.cpu)

    device = accelerator.device
    print('Using device:', device)
    os.makedirs(args.output_dir, exist_ok = True)

    if args.seed is not None:
        set_seed(args.seed)

    make_save_dir(args)

    if args.mode == 'train' and args.wandb:
        wandb.init(project=f"Preliminary-Experiment-FuseMoE-{args.task}", name=args.ck_file_path.split('/')[-2], save_code=True)
    
    if args.mode=='train':
        train_dataset, train_sampler, train_dataloader = data_perpare(args, 'train')
        val_dataset, val_sampler, val_dataloader = data_perpare(args, 'val')
        _, _, test_data_loader = data_perpare(args, 'test')
    elif args.mode=='test':
         _, _, test_data_loader = data_perpare(args, 'test')

    model = MULTCrossModel(args=args,device=device,orig_d_ts=30, orig_reg_d_ts=60, orig_d_txt=768, ts_seq_num=args.tt_max, text_seq_num=args.num_of_notes)
    
    if 'Text' in args.modeltype:
        optimizer= torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n]},
                {'params': [p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.txt_learning_rate}
            ], lr=args.ts_learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)

    if args.mode == 'train':
        model, optimizer, train_dataloader,val_dataloader,test_data_loader = \
        accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, test_data_loader)
    elif args.mode == 'test':
        model, optimizer, test_data_loader = \
        accelerator.prepare(model, optimizer, test_data_loader)

    if args.mode == 'train':
        trainer_irg(model=model, args=args, accelerator=accelerator, train_dataloader=train_dataloader,\
            dev_dataloader=val_dataloader, test_data_loader=test_data_loader, device=device, optimizer=optimizer)
    eval_test(args, model, test_data_loader, device)

    print(f"New maximum memory allocated on GPU: {torch.cuda.max_memory_allocated(device)} bytes")
    print(f'Results saved in:\n{args.ck_file_path}')


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
