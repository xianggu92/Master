# TRACE

This repository is the code release of the paper: **"TRACE: A Temporal Conditional Estimation for Multimodal Time Series Foundation Models"**.

## Environment
```bash
conda create -n Trace python=3.8
source activate Trace
pip install -r requirements.txt
```

## Dataset Download
Please download the MIMIC-IV dataset from the following Google Drive link: [MIMIC-IV](https://drive.google.com/file/d/1da-jMJ6DuYWy-HZloRJVC6DxdO8F457U/view?usp=drive_link)

After downloading, place the dataset under the directory: `dataset`

## Run Multimodal Conditional Diffusion

```bash
cd diffusion
bash train_and_inference.sh
```

1. **Data Merging**: Combines the MIMIC-IV train/val/test sets into a unified format for diffusion model training.

2. **Training and Inference**

4. **Data Splitting**: Splits the imputed dataset into train/val/test sets for MoE fusion.


## Run  MoE Fusion
```bash
cd ../moe_fusion/scripts
bash run_mimiciv_mod2.sh
```

## Load Results
First change the `filepath` in [load_result.py](moe_fusion/scripts/load_result.py), then run
```bash
python load_result.py
```

## Acknowledgement
Part of the implementation and experienmental results are based on the paper: [ FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion](https://arxiv.org/pdf/2402.03226.pdf). We thank all the authors for their contributions.

## Citation
