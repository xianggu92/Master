export CUDA_VISIBLE_DEVICES=3


for (( i=1; i<=5; i++ ))
do
    python -W ignore main_mimiciv.py \
        --num_train_epochs 8 \
        --mode 'train' \
        --modeltype 'TS_CXR_Text' \
        --kernel_size 1 \
        --train_batch_size 2 \
        --eval_batch_size 8 \
        --seed ${i} \
        --gradient_accumulation_steps 16 \
        --ts_learning_rate 0.0004 \
        --txt_learning_rate 0.00002 \
        --notes_order 'Last' \
        --num_of_notes 5 \
        --output_dir "../run/TS_Text" \
        --layers 1 \
        --embed_dim 128 \
        --num_modalities 2 \
        --task 'ihm-48-notes' \
        --file_path '/mnt/data/yihua/master/datasets/mimic-iv/TS_Text' \
        --num_labels 2 \
        --num_heads 8 \
        --embed_time 64 \
        --tt_max 48 \
        --fp16 \
        --irregular_learn_emb_ts 'mTAND'\
        --irregular_learn_emb_text 'mTAND'\
        --irregular_learn_emb_cxr 'mTAND'\
        --irregular_learn_emb_ecg 'mTAND'\
        --cross_method "moe" \
        --gating_function "laplace" \
        --num_of_experts 16 5 \
        --top_k 4 4 \
        --disjoint_top_k 2 \
        --hidden_size 512 \
        --router_type 'joint' \
        --use_balance_loss \
        --balance_loss_coef 0.01 \
        --reg_ts \
        --TS_mixup \
        --mixup_level 'batch' \
        --wandb
done
