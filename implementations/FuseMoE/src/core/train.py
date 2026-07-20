from utils.checkpoint import check_point
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
import warnings 
import wandb
import torch
import pickle
import os
import numpy as np


def eval_test(args, model, test_data_loader, device):
    model.eval()
    rootdir = args.ck_file_path

    os.makedirs(rootdir,exist_ok = True)

    try:
        result_dict = pickle.load(open(rootdir+"result.pkl", "rb"))
    except:
        result_dict={}

    seed = args.seed
    result_dict[seed] = {}
    for subdir, dirs, files in os.walk(rootdir):
        substr = subdir.split('/')[-1]
        if args.monitor not in substr:
            continue

        file = str(seed) + '.pth.tar'
        file_path = os.path.join(subdir, file)
        print(file_path)
        checkpoint = torch.load(file_path, weights_only=False)
        model.load_state_dict(checkpoint['network'])
        test_val = evaluate_irg(args=args, device=device, data_loader=test_data_loader, model=model)
        print(test_val)
        for eval_type, val in test_val.items():
            result_dict[seed][eval_type]={}
            result_dict[seed][eval_type]['val']=checkpoint['best_val'][eval_type]
            result_dict[seed][eval_type]['test']=test_val[eval_type]

    with open(rootdir+"/result.pkl","wb") as f:
        pickle.dump(result_dict, f)


def trainer_irg(model, args, accelerator, train_dataloader, dev_dataloader, test_data_loader, device, optimizer):
    best_evals = {}
    global_step = 0
    early_stopping_counter = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        epoch_total_loss = 0.0
        epoch_loss = 0.0
        epoch_balance_loss = 0.0
        total_samples = 0

        for step, batch in enumerate(tqdm(train_dataloader, ncols=45)):
            global_step += 1

            result = model(**batch)

            if isinstance(result, tuple):
                loss, balance_loss = result
            else:
                loss = result
                balance_loss = None

            if loss is None:
                warnings.warn("loss is None!")
                continue

            # Incorporate balance_loss if enabled and available
            if hasattr(args, 'use_balance_loss') and args.use_balance_loss and balance_loss is not None:
                total_loss = loss + args.balance_loss_coef * balance_loss
            else:
                total_loss = loss

            # 累積 epoch loss
            batch_size = batch["labels"].size(0)
            epoch_total_loss += total_loss.item() * batch_size
            epoch_loss += loss.item() * batch_size
            if balance_loss is not None:
                epoch_balance_loss += balance_loss.item() * batch_size
            total_samples += batch_size

            total_loss = total_loss / args.gradient_accumulation_steps
            accelerator.backward(total_loss)

            if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                if args.gradient_clipping:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                model.zero_grad()

        if args.wandb:
            wandb.log({
                'Epoch': epoch + 1,
                'Global Step': global_step,
                'Task Loss': epoch_loss / total_samples,
                'Balance Loss': epoch_balance_loss / total_samples,
                'Total Loss': epoch_total_loss / total_samples,
            })

        eval_vals = evaluate_irg(args, device, dev_dataloader, model)

        if eval_vals[args.monitor] > best_evals.get(args.monitor, float('-inf')):
            best_evals = eval_vals.copy()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Print current and best metrics
        print(f'Epoch: {epoch+1}')
        for k, v in eval_vals.items():
            if args.wandb: 
                wandb.log({
                    'Epoch': epoch + 1,
                    'Global Step': global_step,
                    f'Val {k}': v, }
                    )
            print(f"Current {k} {v}")
            print(f"Best {k} {best_evals[k]}")

        if early_stopping_counter >= args.patience:
            print('Early stopping')
            break


def evaluate_irg(args, device, data_loader, model):
    model.eval()
    eval_logits = []
    eval_example = []
    for idx, batch in enumerate(tqdm(data_loader, ncols=45)):
        labels = batch.pop('labels')

        with torch.no_grad():
            logits = model(**batch)

            logits = logits.cpu().numpy()
            label_ids = labels.cpu().numpy()
            eval_logits += logits.tolist()
            eval_example += label_ids.tolist()

    eval_vals={}
    all_logits = np.array(eval_logits)
    all_label = np.array(eval_example)
    all_pred= np.where(all_logits > 0.5, 1, 0)

    if 'pheno' in args.task:
        eval_vals['auroc'] = roc_auc_score(np.array(eval_example), np.array(eval_logits), average="macro")
        eval_vals['auprc'] = average_precision_score(np.array(eval_example), np.array(eval_logits), average='macro')
        eval_vals['f1'] = f1_score(all_label, all_pred, average='macro')
        eval_vals['recall'] = recall_score(np.array(eval_example), all_pred, average='macro')
        eval_vals['precision'] = precision_score(np.array(eval_example), all_pred, average='macro')

        check_point(eval_vals, model, eval_logits, args, args.monitor)

    elif 'ihm' in args.task or 'los' in args.task:
        eval_vals['auroc'] = roc_auc_score(np.array(eval_example), np.array(eval_logits))
        eval_vals['auprc'] = average_precision_score(np.array(eval_example), np.array(eval_logits))
        eval_vals['f1'] = f1_score(np.array(eval_example), all_pred)
        eval_vals['recall'] = recall_score(np.array(eval_example), all_pred)
        eval_vals['precision'] = precision_score(np.array(eval_example), all_pred)

        check_point(eval_vals, model, eval_logits, args, args.monitor)

    return eval_vals
