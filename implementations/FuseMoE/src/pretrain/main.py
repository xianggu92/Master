import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pretrain.dataset import data_perpare
from pretrain.model import DeltaPredictor
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import set_seed
import wandb


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_samples = 0
    total_loss = 0
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        label = batch.pop('label')

        preds = model(**batch)
        
        loss = criterion(preds, label)
        masked_loss = (loss * batch['mask']) / batch['mask'].sum()
        
        optimizer.zero_grad()
        masked_loss.backward()
        optimizer.step()
        
        current_batch_size = batch['x'].shape[0]
        total_samples += current_batch_size
        total_loss += loss.item() * current_batch_size
        
    return total_loss / total_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_samples = 0
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            label = batch.pop('label')
            
            preds = model(**batch)

            loss = criterion(preds, label)
            masked_loss = (loss * batch['mask']) / batch['mask'].sum()
            
            current_batch_size = batch['x'].shape[0]
            total_samples += current_batch_size
            total_loss += masked_loss.item() * current_batch_size
            
    return total_loss / total_samples


def main(args):
    set_seed(args.seed)
    device = args.device
    args.ck_file_name = f'{args.task}_seed{args.seed}_ep{args.epochs}_bs{args.train_batch_size}_lr{args.lr}_rp{args.n_ref_point}_et{args.embed_time}_hidden{args.nhidden}_head{args.num_heads}'

    if args.wandb:
        wandb.init(project=f"Pretrain-mTAND-{args.task}", name=args.ck_file_path.split('/')[-2])
    
    _, _, train_dataloader = data_perpare(args, 'train')
    _, _, val_dataloader = data_perpare(args, 'val')
    _, _, test_dataloader = data_perpare(args, 'test')
    
    model = DeltaPredictor(
        input_dim=args.input_dim, 
        n_ref_point=args.n_ref_point,
        latent_dim=args.latent_dim,
        nhidden=args.nhidden,
        embed_time=args.embed_time,
        num_heads=args.num_heads
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='none')
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        val_loss = evaluate(model, val_dataloader, criterion, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, args.task + '.pth'))

        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    test_loss = evaluate(model, test_dataloader, criterion, device)


if __name__=='__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--file_path", type=str, default="Data", help="A path to dataset folder")
    parser.add_argument("--task", type=str, default="ihm")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size  for the training dataloader.")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for the evaluation dataloader.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--debug", action="store_true", help="Debug mode with less data")
    parser.add_argument("--tt_max", type=int, default=48, help="Max time duration of the time series")
    parser.add_argument("--wandb", action='store_true')
    
    # 模型超參數
    parser.add_argument("--input_dim", type=int, default=30, help="Number of features in time series")
    parser.add_argument("--n_ref_point", type=int, default=8, help="Number of reference points for mTAN")
    parser.add_argument("--nhidden", type=int, default=16)
    parser.add_argument("--embed_time", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=1)
    
    args = parser.parse_known_args()[0]

    main()