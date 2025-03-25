#!/bin/bash

deepspeed --include localhost:1,2 ./transformers/examples/pytorch/language-modeling/run_clm.py  \
    --deepspeed ds_config.json \
    --model_name_or_path q_k_iter1_rm \
    --train_file /home/cho/finetune/iter2_rm_data/train_prompt.txt \
    --validation_file /home/cho/finetune/iter2_rm_data/train_prompt.txt \
    --do_train \
    --do_eval \
    --num_train_epochs 3 \
    --save_steps 5000 \
    --save_total_limit 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --output_dir q_k_iter2_rm/
    
#初回学習時のパラメタ上書き設定(2回目以降はローカルに保存したパラメタを利用)
    # --config_overrides="vocab_size=50000,bos_token_id=1,eos_token_id=2,n_embd=1600,n_layer=48,n_head=20" \
