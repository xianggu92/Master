
import os
import torch
import numpy as np
import argparse
import random
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, recall_score, precision_score
from copy import deepcopy
from tqdm import trange
from models import FlexMoE
from utils import seed_everything, setup_logger
from data import load_and_preprocess_data, create_loaders, data_prepare
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork()")

# Utility function to convert string to bool
def str2bool(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return (s == 'True') or (s == 'true')

# Parse input arguments
def parse_args():
    parser = argparse.ArgumentParser(description='FlexMoE')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--data', type=str, default='adni')
    parser.add_argument('--modality', type=str, default='IGCB') # I G C B for ADNI, L N C for MIMIC
    parser.add_argument('--preprocessed', type=str2bool, default=True) # Whether to use preprocessed image modality
    parser.add_argument('--initial_filling', type=str, default='mean') # None mean
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--warm_up_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--top_k', type=int, default=4) # Number of Routers
    parser.add_argument('--num_patches', type=int, default=16) # Number of Patches for Input Token
    parser.add_argument('--num_experts', type=int, default=16) # Number of Experts
    parser.add_argument('--num_routers', type=int, default=1) # Number of Routers
    parser.add_argument('--num_layers_enc', type=int, default=1) # Number of MLP layers for encoders
    parser.add_argument('--num_layers_fus', type=int, default=1) # Number of MLP layers for fusion model
    parser.add_argument('--num_layers_pred', type=int, default=1) # Number of MLP layers for prediction head
    parser.add_argument('--num_heads', type=int, default=4) # Number of heads
    parser.add_argument('--num_workers', type=int, default=4) # Number of workers for DataLoader
    parser.add_argument('--pin_memory', type=str2bool, default=True) # Pin memory in DataLoader
    parser.add_argument('--use_common_ids', type=str2bool, default=False) # Use common ids across modalities    
    parser.add_argument('--dropout', type=float, default=0.5) # Number of Routers
    parser.add_argument('--gate_loss_weight', type=float, default=1e-2)
    parser.add_argument('--save', type=str2bool, default=True)
    parser.add_argument('--load_model', type=str2bool, default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--task', type=str, default='ihm-48-cxr-notes-ecg')
    parser.add_argument('--debug', action='store_true')

    return parser.parse_known_args()

def run_epoch(args, loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, is_training=False, optimizer=None, gate_loss_weight=0.0):
    all_preds = []
    all_labels = []
    all_probs = []
    task_losses = []
    gate_losses = []
    
    if is_training:
        fusion_model.train()
        for encoder in encoder_dict.values():
            encoder.train()
    else:
        fusion_model.eval()
        for encoder in encoder_dict.values():
            encoder.eval()

    for batch_samples, batch_labels, batch_mcs, batch_observed in loader:
        batch_samples = {k: v.to(device, non_blocking=True) for k, v in batch_samples.items()}
        batch_labels = batch_labels.to(device, non_blocking=True)
        batch_mcs = batch_mcs.to(device, non_blocking=True)
        batch_observed = batch_observed.to(device, non_blocking=True)
        
        fusion_input = []
        for i, (modality, samples) in enumerate(batch_samples.items()):
            mask = batch_observed[:, modality_dict[modality]]
            encoded_samples = torch.zeros((samples.shape[0], args.num_patches, args.hidden_dim)).to(device)
            if mask.sum() > 0:
                encoded_samples[mask] = encoder_dict[modality](samples[mask])
            if (~mask).sum() > 0:
                encoded_samples[~mask] = missing_embeds[batch_mcs[~mask], modality_dict[modality]]
            fusion_input.append(encoded_samples)

        outputs = fusion_model(*fusion_input, expert_indices=batch_mcs)

        if is_training:
            optimizer.zero_grad()
            task_loss = criterion(outputs, batch_labels)
            task_losses.append(task_loss.item())
            gate_loss = fusion_model.gate_loss()
            gate_losses.append(float(gate_loss))
            loss = task_loss + gate_loss_weight * gate_loss
            loss.backward()
            optimizer.step()
        else:
            all_labels.extend(batch_labels.cpu().numpy())
            all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy())

    if is_training:
        return task_losses, gate_losses
    else:
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = all_probs > 0.5
        return all_preds, all_labels, all_probs


def train_and_evaluate(args, seed, save_path=None):
    seed_everything(seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    num_modalities = len(args.modality)

    if args.data == 'adni':
        modality_dict = {'image':0, 'genomic': 1, 'clinical': 2, 'biospecimen': 3}
        args.n_full_modalities = len(modality_dict)
        data_dict, encoder_dict, labels, train_ids, valid_ids, test_ids, n_labels, input_dims, transforms, masks, observed_idx_arr, full_modality_index = load_and_preprocess_data(args, modality_dict)
        train_loader, train_loader_shuffle, val_loader, test_loader = create_loaders(data_dict, observed_idx_arr, labels, train_ids, valid_ids, test_ids, args.batch_size, args.num_workers, args.pin_memory, input_dims, transforms, masks, args.preprocessed, args.use_common_ids)
    elif args.data == 'mimic_iv':
        modality_dict = {'ts_feats':0, 'text_feats': 1, 'cxr_feats': 2, 'ecg_feats': 3}
        args.n_full_modalities = len(modality_dict)
        full_modality_index = 0
        n_labels = 2 if 'ihm' or 'los' in args.task else 25
        _, _, train_loader, train_loader_shuffle, encoder_dict = data_prepare(args, 'train')
        _, _, _, val_loader, _ = data_prepare(args, 'val')
        _, _, _, test_loader, _ = data_prepare(args, 'test')
        
    fusion_model = FlexMoE(num_modalities, full_modality_index, args.num_patches, args.hidden_dim, n_labels, args.num_layers_fus, args.num_layers_pred, args.num_experts, args.num_routers, args.top_k, args.num_heads, args.dropout).to(device)
    params = list(fusion_model.parameters()) + [param for encoder in encoder_dict.values() for param in encoder.parameters()]    
    if num_modalities > 1:
        missing_embeds = torch.nn.Parameter(torch.randn((2**num_modalities)-1, args.n_full_modalities, args.num_patches, args.hidden_dim, dtype=torch.float, device=device), requires_grad=True)
        params += [missing_embeds]

    optimizer = torch.optim.Adam(params, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss() # if args.data == 'adni' else torch.nn.CrossEntropyLoss(torch.tensor([0.25, 0.75]).to(device))

    best_val_auprc = 0.0

    if save_path is None:
        for epoch in trange(args.train_epochs):
            fusion_model.train()
            for encoder in encoder_dict.values():
                encoder.train()

            if epoch >= args.warm_up_epochs:
                train_loader_new = train_loader_shuffle
                warm_up_tag = ''
                train_epochs = args.train_epochs
            else:
                # activate modality-based sorting
                train_loader_new = train_loader
                warm_up_tag = 'Warm Up ' 
                train_epochs = args.warm_up_epochs

            ## Training
            task_losses, gate_losses = run_epoch(args, train_loader_new, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device, is_training=True, optimizer=optimizer, gate_loss_weight=args.gate_loss_weight)
            
            ## Validation
            fusion_model.eval()
            for encoder in encoder_dict.values():
                encoder.eval()
            with torch.no_grad():
                val_preds, val_labels, val_probs = run_epoch(args, val_loader, encoder_dict, modality_dict, missing_embeds, fusion_model, criterion, device)
            val_auroc = roc_auc_score(val_labels, val_probs[:, 1], average='macro') if n_labels == 2 else roc_auc_score(val_labels, val_probs, multi_class='ovr', average='macro')
            val_auprc = average_precision_score(val_labels,  val_probs[:, 1], average='macro') if n_labels == 2 else average_precision_score(val_labels,  val_probs, multi_class='ovr', average='macro')
            val_f1 = f1_score(val_labels, val_preds[:, 1], average='macro') if n_labels == 2 else f1_score(val_labels, val_preds, multi_class='ovr', average='macro')
            val_recall = recall_score(val_labels, val_preds[:, 1], average='macro') if n_labels == 2 else recall_score(val_labels, val_preds, multi_class='ovr', average='macro')
            val_precision = precision_score(val_labels, val_preds[:, 1], average='macro') if n_labels == 2 else precision_score(val_labels, val_preds, multi_class='ovr', average='macro')

            if val_auprc > best_val_auprc:
                print(f" [(**Best**) {warm_up_tag}Epoch {epoch+1}/{train_epochs}] Val AUROC: {val_auroc*100:.2f}, Val AUPRC: {val_auprc*100:.2f}, Val F1: {val_f1*100:.2f}, Val Recall: {val_recall*100:.2f}, Val Precision: {val_precision*100:.2f}")
                best_val_auroc = val_auroc
                best_val_auprc = val_auprc
                best_val_f1 = val_f1
                best_val_recall = val_recall
                best_val_precision = val_precision
                best_model_me = deepcopy(missing_embeds)
                best_model_fus = deepcopy(fusion_model)
                best_model_enc = deepcopy(encoder_dict)

            print(f"[Seed {seed}/{args.n_runs-1}] [{warm_up_tag}Epoch {epoch+1}/{train_epochs}] Task Loss: {np.mean(task_losses):.2f}, Router Loss: {np.mean(gate_losses):.2f} / Val AUROC: {val_auroc*100:.2f}, Val AUPRC: {val_auprc*100:.2f}, Val F1: {val_f1*100:.2f}, Val Recall: {val_recall*100:.2f}, Val Precision: {val_precision*100:.2f}")

        # Save the best model
        if args.save:
            os.makedirs('./saves', exist_ok=True)
            save_path = f'./saves/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}.pth'
            torch.save({
                'missing_embeds': best_model_me,
                'fusion_model': best_model_fus.state_dict(),
                'encoder_dict': {modality: deepcopy(encoder.state_dict()) for modality, encoder in best_model_enc.items()}
            }, save_path)

            print(f"Best model saved to {save_path}")
    
    else:
        best_model_me = missing_embeds
        best_model_fus = fusion_model
        best_model_enc = encoder_dict

        # Load the saved model onto the correct device (GPU or CPU)
        checkpoint = torch.load(save_path, map_location=device)

        # Load the models' states
        best_model_me = checkpoint['missing_embeds']
        best_model_fus.load_state_dict(checkpoint['fusion_model'])
        for modality, encoder in best_model_enc.items():
            encoder.load_state_dict(checkpoint['encoder_dict'][modality])
            encoder.to(device)
            encoder.eval()

        # Move the models to the correct device if necessary
        best_model_me.to(device)
        best_model_fus.to(device)

        ## Validation
        with torch.no_grad():
            val_preds, val_labels, val_probs = run_epoch(args, val_loader, best_model_enc, modality_dict, best_model_me, best_model_fus, criterion, device)
        best_val_auroc = val_auroc
        best_val_auprc = val_auprc
        best_val_f1 = val_f1
        best_val_recall = val_recall
        best_val_precision = val_precision

    ## Test
    with torch.no_grad():
        test_preds, test_labels, test_probs = run_epoch(args, test_loader, best_model_enc, modality_dict, best_model_me, best_model_fus, criterion, device)
    test_auroc = roc_auc_score(test_labels, test_probs[:, 1], average='macro') if n_labels == 2 else roc_auc_score(test_labels, test_probs, multi_class='ovr', average='macro')
    test_auprc = average_precision_score(test_labels,  test_probs[:, 1], average='macro') if n_labels == 2 else average_precision_score(test_labels,  test_probs, multi_class='ovr', average='macro')
    test_f1 = f1_score(test_labels, test_preds[:, 1], average='macro') if n_labels == 2 else f1_score(test_labels, test_preds, multi_class='ovr', average='macro')
    test_recall = recall_score(test_labels, test_preds[:, 1], average='macro') if n_labels == 2 else recall_score(test_labels, test_preds, multi_class='ovr', average='macro')
    test_precision = precision_score(test_labels, test_preds[:, 1], average='macro') if n_labels == 2 else precision_score(test_labels, test_preds, multi_class='ovr', average='macro')

    return best_val_auroc, best_val_auprc, best_val_f1, best_val_recall, best_val_precision, test_auroc, test_auprc, test_f1, test_recall, test_precision

def main():
    args, _ = parse_args()
    logger = setup_logger('./logs', f'{args.data}', f'{args.modality}.txt')
    seeds = np.arange(args.n_runs) # [0, 1, 2]
    val_aurocs = []
    val_auprcs = []
    val_f1s = []
    val_recalls = []
    val_precisions = []
    test_aurocs = []
    test_auprcs = []
    test_f1s = []
    test_recalls = []
    test_precisions = []
    
    log_summary = "======================================================================================\n"
    
    model_kwargs = {
        "model": 'FlexMoE',
        "modality": args.modality,
        "initial_filling": args.initial_filling,
        "use_common_ids": args.use_common_ids,
        "train_epochs": args.train_epochs,
        "warm_up_epochs": args.warm_up_epochs,
        "num_experts": args.num_experts,
        "num_routers": args.num_routers,
        "top_k": args.top_k,
        "num_layers_enc": args.num_layers_enc,
        "num_layers_fus": args.num_layers_fus,
        "num_layers_pred": args.num_layers_pred,
        "num_heads": args.num_heads,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "num_patches": args.num_patches,
        "gate_loss_weight": args.gate_loss_weight,
    }

    log_summary += f"Model configuration: {model_kwargs}\n"

    print('Modality:', args.modality)

    for seed in seeds:
        if (not args.save) & (args.load_model):
            save_path = f'./saves/seed_{seed}_modality_{args.modality}_train_epochs_{args.train_epochs}.pth'
        else:
            save_path = None
        val_auroc, val_auprc, val_f1, val_recall, val_precision, test_auroc, test_auprc, test_f1, test_recall, test_precision = train_and_evaluate(args, seed, save_path=save_path)
        val_aurocs.append(val_auroc)
        val_auprcs.append(val_auprc)
        val_f1s.append(val_f1)
        val_recalls.append(val_recall)
        val_precisions.append(val_precision)
        test_aurocs.append(test_auroc)
        test_auprcs.append(test_auprc)
        test_f1s.append(test_f1)
        test_recalls.append(test_recall)
        test_precisions.append(test_precision)
    
    val_avg_auroc = np.mean(val_aurocs) * 100
    val_std_auroc = np.std(val_aurocs) * 100
    val_avg_auprc = np.mean(val_auprcs) * 100
    val_std_auprc = np.std(val_auprcs) * 100
    val_avg_f1 = np.mean(val_f1s) * 100
    val_std_f1 = np.std(val_f1s) * 100
    val_avg_recall = np.mean(val_recalls) * 100
    val_std_recall = np.std(val_recalls) * 100
    val_avg_precision = np.mean(val_precisions) * 100
    val_std_precision = np.std(val_precisions) * 100


    test_avg_auroc = np.mean(test_aurocs) * 100
    test_std_auroc = np.std(test_aurocs) * 100
    test_avg_auprc = np.mean(test_auprcs) * 100
    test_std_auprc = np.std(test_auprcs) * 100
    test_avg_f1 = np.mean(test_f1s) * 100
    test_std_f1 = np.std(test_f1s) * 100
    test_avg_recall = np.mean(test_recalls) * 100
    test_std_recall = np.std(test_recalls) * 100
    test_avg_precision = np.mean(test_precisions) * 100
    test_std_precision = np.std(test_precisions) * 100

    log_summary += f'[Val] Avg AUROC: {val_avg_auroc:.2f} ± {val_std_auroc:.2f} | '
    log_summary += f'AUPRC: {val_avg_auprc:.2f} ± {val_std_auprc:.2f} | '
    log_summary += f'F1: {val_avg_f1:.2f} ± {val_std_f1:.2f} | '
    log_summary += f'Recall: {val_avg_recall:.2f} ± {val_std_recall:.2f} | '
    log_summary += f'Precision: {val_avg_precision:.2f} ± {val_std_precision:.2f} / '

    log_summary += f'[Test] Avg AUROC: {test_avg_auroc:.2f} ± {test_std_auroc:.2f} | '
    log_summary += f'AUPRC: {test_avg_auprc:.2f} ± {test_std_auprc:.2f} | '
    log_summary += f'F1: {test_avg_f1:.2f} ± {test_std_f1:.2f} | '
    log_summary += f'Recall: {test_avg_recall:.2f} ± {test_std_recall:.2f} | '
    log_summary += f'Precision: {test_avg_precision:.2f} ± {test_std_precision:.2f} '

    print(model_kwargs)
    print(
        f'[Val] Avg AUROC: {val_avg_auroc:.2f} ± {val_std_auroc:.2f} / '
        f'AUPRC: {val_avg_auprc:.2f} ± {val_std_auprc:.2f} / '
        f'F1: {val_avg_f1:.2f} ± {val_std_f1:.2f} / '
        f'Recall: {val_avg_recall:.2f} ± {val_std_recall:.2f} / '
        f'Precision: {val_avg_precision:.2f} ± {val_std_precision:.2f}'
    )
    print(
        f'[Test] Avg AUROC: {test_avg_auroc:.2f} ± {test_std_auroc:.2f} / '
        f'AUPRC: {test_avg_auprc:.2f} ± {test_std_auprc:.2f} / '
        f'F1: {test_avg_f1:.2f} ± {test_std_f1:.2f} / '
        f'Recall: {test_avg_recall:.2f} ± {test_std_recall:.2f} / '
        f'Precision: {test_avg_precision:.2f} ± {test_std_precision:.2f}'
    )

    logger.info(log_summary)

if __name__ == '__main__':
    main()