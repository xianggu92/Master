from utils.checkpoint import *
from utils.util import *
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score
import warnings 
import wandb


def eval_test(args, model, test_data_loader, device):
    model.eval()
    rootdir = args.ck_file_path

    os.makedirs(rootdir,exist_ok = True)

    try:
        result_dict = pickle.load(open(rootdir+"result.pkl", "rb"))
    except:
        result_dict={}

    if args.generate_data:
        seed = str(args.datagereate_seed)+'_'+str(args.seed)
    else:
        seed = args.seed
    result_dict[seed] = {}
    for subdir, dirs, files in os.walk(rootdir):
        substr = subdir.split('/')[-1]
        if 'f1' not in substr:
            continue

        file = str(seed) + '.pth.tar'
        file_path = os.path.join(subdir, file)
        print(file_path)
        checkpoint = torch.load(file_path)
        model.load_state_dict(checkpoint['network'])
        test_val = evaluate_irg(args=args, device=device, data_loader=test_data_loader, model=model)
        print(test_val)
        for eval_type, val in test_val.items():
            result_dict[seed][eval_type]={}
            # result_dict[seed][eval_type]['val']=checkpoint['best_val'][eval_type]
            result_dict[seed][eval_type]['test']=test_val[eval_type]

    with open(rootdir+"/result.pkl","wb") as f:
        pickle.dump(result_dict, f)


def trainer_irg(model, args, accelerator, train_dataloader, dev_dataloader, test_data_loader, device, optimizer):
    best_evals = {}
    global_step = 0
    for epoch in range(args.num_train_epochs):
        model.train()

        for step, batch in enumerate(tqdm(train_dataloader)):
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

            total_loss = total_loss / args.gradient_accumulation_steps
            accelerator.backward(total_loss)

            if (step+1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                model.zero_grad()

            if args.wandb:
                wandb.log({
                    'Epoch': epoch,
                    'Global Step': global_step,
                    'Train Loss': loss,
                    'Balance Loss': balance_loss,
                    'Total Loss': total_loss,
                })

        eval_vals = evaluate_irg(args, device, dev_dataloader, model)

        print("Epoch: " + str(epoch+1))
        for k, v in eval_vals.items():
            if k == 'auc_scores':
              continue
            if args.wandb:
                wandb.log({
                    'Epoch': epoch,
                    'Global Step': global_step,
                    f'Val {k}': v,
                })
            best_eval = best_evals.get(k, 0)
            if v > best_eval:
                best_eval = v
                best_evals[k] = best_eval
            print("Current " + k + ' ' + str(v))
            print("Best " + k + ' ' + str(best_eval))


def evaluate_irg(args, device, data_loader, model):
    model.eval()
    eval_logits = []
    eval_example = []
    for idx, batch in enumerate(tqdm(data_loader)):
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
        eval_vals = metrics_multilabel(all_label, all_logits, verbose=0)
        eval_vals['macro_f1'] = f1_score(all_label, all_pred, average='macro')
        eval_vals['macro_precision'] = precision_score(np.array(eval_example), all_pred, average='macro')
        eval_vals['macro_recall'] = recall_score(np.array(eval_example), all_pred, average='macro')

        check_point(eval_vals, model, eval_logits, args, "macro_f1")

    elif 'ihm' in args.task or 'los' in args.task:
        eval_val = roc_auc_score(np.array(eval_example), np.array(eval_logits))
        eval_vals['auc'] = eval_val
        (precisions, recalls, thresholds) = precision_recall_curve(np.array(eval_example), np.array(eval_logits))
        eval_val = auc(recalls, precisions)
        eval_vals['auprc'] = eval_val
        eval_val = f1_score(np.array(eval_example), all_pred)
        eval_vals['f1'] = eval_val
        eval_vals['recall'] = recall_score(np.array(eval_example), all_pred)
        eval_vals['precision'] = precision_score(np.array(eval_example), all_pred)

        check_point(eval_vals, model, eval_logits, args, "f1")

    return eval_vals
