import os
import argparse
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_mm import MultiModalImputationDataset, collate_mm, load_samples_cached
from csdi_mm_moe import CSDI_MultiModal_MoE


@torch.no_grad()
def impute_batch_full(model, batch, n_samples: int = 20):

    model.eval()

    (observed_data, observed_mask, observed_tp,
     gt_mask, hist_mask, seq_len,
     text_vec, cxr_vec, ecg_vec,
     text_missing, cxr_missing, ecg_missing) = model.process_data(batch)

    cond_mask = observed_mask

    fused_ctx, moe_w = model.fuse_ctx(
        text_vec, cxr_vec, ecg_vec,
        text_missing, cxr_missing, ecg_missing
    )
    side_info = model.get_side_info_mm(observed_tp, cond_mask, fused_ctx)

    samples = model.impute(observed_data, cond_mask, side_info, n_samples)
    pred = samples.median(dim=1).values  # (B, K, L)

    imputed = observed_data * observed_mask + pred * (1.0 - observed_mask)
    return imputed, observed_mask, seq_len, moe_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    # output control
    parser.add_argument("--output_path", type=str, default=None, help="full output file path (.pkl)")
    parser.add_argument("--output_dir", type=str, default=None, help="if set, output_path = output_dir/output_name")
    parser.add_argument("--output_name", type=str, default="imputed_dataset.pkl")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--tiny", type=int, default=None)

    # cache
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    # resolve output path
    if args.output_path is None:
        if args.output_dir is None:
            raise ValueError("provide --output_path or (--output_dir and optionally --output_name)")
        os.makedirs(args.output_dir, exist_ok=True)
        args.output_path = os.path.join(args.output_dir, args.output_name)

    # device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print("device:", device)

    # load checkpoint
    ckpt = torch.load(args.model_path, map_location="cpu")
    config = ckpt["config"]
    num_experts = int(config.get("multimodal", {}).get("num_experts", 3))
    print(f"num_experts(from ckpt config) = {num_experts}")

    # load data (cache)
    samples = load_samples_cached(
        args.data_path,
        cache_dir=args.cache_dir,
        use_cache=(not args.no_cache),
        verbose=True,
    )
    if args.tiny is not None:
        samples = samples[:args.tiny]
        print(f"[tiny] using first {len(samples)} samples")

    # dataset / dataloader
    ds = MultiModalImputationDataset(
        samples,
        text_dim=config["data"]["text_dim"],
        ecg_dim=config["data"]["ecg_dim"],
        cxr_dim=config["data"]["cxr_dim"],
        eval_mask_ratio=0.0,
        seed=config["train"]["seed"],
        mode="test",
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_mm,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    # model
    model = CSDI_MultiModal_MoE(config=config, device=device, target_dim=30).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    out_samples = samples
    seen = 0

    pbar = tqdm(dl, desc="imputing", leave=True)
    for batch in pbar:
        imputed, observed_mask, seq_len, moe_w = impute_batch_full(
            model, batch, n_samples=args.n_samples
        )

        imputed = imputed.cpu().numpy()       # (B,K,Lpad)
        seq_len = seq_len.cpu().numpy()
        moe_w = moe_w.cpu().numpy()           # (B,E)

        B = imputed.shape[0]
        for i in range(B):
            idx = seen + i
            L = int(seq_len[i])

            imputed_i = imputed[i, :, :L].T  # (L,K)

            out_samples[idx]["irg_ts_imputed"] = imputed_i.astype(np.float32)
            out_samples[idx]["moe_w_experts"] = moe_w[i].astype(np.float32)  # length = num_experts
            out_samples[idx]["num_experts"] = int(num_experts)

        seen += B

    assert seen == len(out_samples), f"processed {seen} != {len(out_samples)}"

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(out_samples, f)

    print(f"Saved imputed dataset to: {args.output_path}")


if __name__ == "__main__":
    main()
