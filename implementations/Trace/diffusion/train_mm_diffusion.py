import os
import argparse
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt

from config_mm import get_config
from dataset_mm import MultiModalImputationDataset, collate_mm, load_samples_cached
from csdi_mm_moe import CSDI_MultiModal_MoE
from reproducibility import configure_reproducibility


@torch.no_grad()
def eval_one_epoch_diffusion_mse(model, loader, desc="val"):
    """Compute diffusion noise-prediction MSE on the validation target region."""
    model.eval()
    weighted_loss_sum = 0.0
    total_target_count = 0.0

    pbar = tqdm(loader, desc=f"[{desc}] diffusion loss", leave=False)
    for batch in pbar:
        loss, _ = model(batch, is_train=0)
        target_count = (
            batch["irg_ts_mask"].float() - batch["gt_mask"].float()
        ).clamp_min(0).sum().item()

        weighted_loss_sum += loss.item() * target_count
        total_target_count += target_count

        running_loss = weighted_loss_sum / max(total_target_count, 1.0)
        pbar.set_postfix(mse=f"{running_loss:.4f}")

    return float(weighted_loss_sum / max(total_target_count, 1.0))


def save_diffusion_mse_plot(out_png: str, eval_epochs, val_diffusion_mse_epoch):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.figure()
    plt.plot(eval_epochs, val_diffusion_mse_epoch, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Diffusion MSE Loss")
    plt.title("Validation Diffusion MSE Loss vs Epoch")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="path to samples (.pt/.pkl/.npy), list[dict]")

    # training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None, help="override the config RNG seed")

    # device / loader
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--tiny", type=int, default=None)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    # evaluation
    parser.add_argument("--val_every", type=int, default=10, help="evaluate every N epochs (default: 10)")

    # MoE ablation
    parser.add_argument("--num_experts", type=int, default=None, help="concat->experts MoE expert count")

    # outputs
    parser.add_argument("--metrics_root", type=str, default="metrics", help="root folder to store CSV/plots")
    parser.add_argument("--folder_name", type=str, default=None, help="subfolder name for this run")
    parser.add_argument("--ckpt_path", type=str, default=None, help="where to save best ckpt (.pt)")

    # cache
    parser.add_argument("--cache_dir", type=str, default=None, help="cache directory for parsed samples")
    parser.add_argument("--no_cache", action="store_true", help="disable cache even if cache_dir is set")

    args = parser.parse_args()

    config = get_config()

    # override config from CLI
    if args.epochs is not None:
        config["train"]["max_epochs"] = args.epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.seed is not None:
        config["train"]["seed"] = int(args.seed)

    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
    else:
        if int(config["train"].get("num_workers", 0)) == 0:
            config["train"]["num_workers"] = 4

    if args.num_experts is not None:
        config.setdefault("multimodal", {})
        config["multimodal"]["num_experts"] = int(args.num_experts)

    num_experts = int(config["multimodal"].get("num_experts", 3))

    seed = configure_reproducibility(config["train"]["seed"])
    print(f"reproducibility seed: {seed} (deterministic algorithms enabled)")

    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    # output dir named by expert count
    if args.folder_name is None:
        exp_dir = ensure_dir(os.path.join(args.metrics_root, f"experts_{num_experts}"))
    else:
        exp_dir = ensure_dir(os.path.join(args.metrics_root, args.folder_name))
    plots_dir = ensure_dir(os.path.join(exp_dir, "plots"))
    csv_path = os.path.join(exp_dir, "metrics.csv")
    diffusion_mse_png = os.path.join(plots_dir, "val_diffusion_mse.png")

    # ckpt path
    if args.ckpt_path is None:
        ensure_dir(os.path.join(exp_dir, "ckpt"))
        args.ckpt_path = os.path.join(exp_dir, "ckpt", "best.pt")
    ensure_dir(os.path.dirname(args.ckpt_path) or ".")
    print("Checkpoint will be saved to:", args.ckpt_path)
    print("Metrics will be saved to:", csv_path)
    print("Plots will be saved to:", plots_dir)

    # load samples (cache)
    samples = load_samples_cached(
        args.data,
        cache_dir=args.cache_dir,
        use_cache=(not args.no_cache),
        verbose=True,
    )
    if args.tiny is not None:
        samples = samples[: args.tiny]
        print(f"[tiny] using first {len(samples)} samples")

    n = len(samples)
    assert n >= 10, f"dataset too small: {n}"

    # 75/15/15 split
    n_train = int(n * 0.75)
    n_val = max(1, int(n * 0.15))
    n_train = max(1, n_train)
    if n_train + n_val >= n:
        n_train = max(1, n - n_val - 1)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    print(f"split: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)}")
    print(f"num_experts={num_experts}")

    ds_train = MultiModalImputationDataset(
        train_samples,
        text_dim=config["data"]["text_dim"],
        ecg_dim=config["data"]["ecg_dim"],
        cxr_dim=config["data"]["cxr_dim"],
        eval_mask_ratio=config["data"]["eval_mask_ratio"],
        seed=config["train"]["seed"],
        mode="train",
    )
    ds_val = MultiModalImputationDataset(
        val_samples,
        text_dim=config["data"]["text_dim"],
        ecg_dim=config["data"]["ecg_dim"],
        cxr_dim=config["data"]["cxr_dim"],
        eval_mask_ratio=config["data"]["eval_mask_ratio"],
        seed=config["train"]["seed"] + 1,
        mode="val",
    )

    use_cuda = (device.type == "cuda")
    num_workers = int(config["train"]["num_workers"])

    dl_train = DataLoader(
        ds_train,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_mm,
        drop_last=True,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(args.prefetch_factor if num_workers > 0 else None),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_mm,
        drop_last=False,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(args.prefetch_factor if num_workers > 0 else None),
    )

    model = CSDI_MultiModal_MoE(config=config, device=device, target_dim=30).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    # prepare CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_diffusion_mse", "num_experts"])

    best_val = float("inf")
    val_diffusion_mse_epoch = []
    eval_epochs = []

    for epoch in range(1, config["train"]["max_epochs"] + 1):
        model.train()
        epoch_losses = []

        pbar = tqdm(dl_train, desc=f"[train] epoch {epoch}/{config['train']['max_epochs']}", leave=True)
        for batch in pbar:
            optim.zero_grad(set_to_none=True)

            loss, moe_w = model(batch, is_train=1)
            loss.backward()

            if config["train"]["grad_clip"] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])

            optim.step()

            loss_item = float(loss.item())
            epoch_losses.append(loss_item)
            pbar.set_postfix(loss=f"{loss_item:.4f}")

        avg_epoch_loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else float("nan")

        do_val = ((epoch % args.val_every) == 0) or (epoch == 1)
        if do_val:
            val_diffusion_mse = eval_one_epoch_diffusion_mse(model, dl_val, desc="val")
            val_diffusion_mse_epoch.append(float(val_diffusion_mse))
            eval_epochs.append(int(epoch))

            print(
                f"[epoch {epoch}] train_loss={avg_epoch_loss:.6f} "
                f"val_diffusion_mse={val_diffusion_mse:.6f}"
            )

            # append CSV row
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch, avg_epoch_loss, val_diffusion_mse, num_experts])

            # save plot every val
            save_diffusion_mse_plot(diffusion_mse_png, eval_epochs, val_diffusion_mse_epoch)

            if val_diffusion_mse < best_val:
                best_val = val_diffusion_mse
                ckpt = {
                    "model": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "val_diffusion_mse": val_diffusion_mse,
                    "num_experts": num_experts,
                }
                torch.save(ckpt, args.ckpt_path)
                print(f"  saved best -> {args.ckpt_path}")
        else:
            print(f"[epoch {epoch}] train_loss={avg_epoch_loss:.6f} (skip val, val_every={args.val_every})")

    print("training done. best_val_diffusion_mse=", best_val)
    print("metrics dir:", exp_dir)


if __name__ == "__main__":
    main()
