#!/bin/bash

RETEST_DATASETS="imagenet cub eurosat food101 imagenetv2 places365 pets dtd flowers102 fgvcaircraft cars"

# Contrastive Adapter (replicate 1), Target classes: Waterbirds
python main.py --dataset waterbirds --torch_seed --load_base_model clip_ViTL14 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layers 1 --hidden_dim 128 --max_epochs 100 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100 --temperature 0.1 --train_sample_ratio 1 --group_size 500 --seed 0 --replicate 1 --avg_thshd 95 --wg_thshd 85
for d in $RETEST_DATASETS; do
    python main_retest.py --dataset waterbirds --torch_seed --load_base_model clip_ViTL14 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layers 1 --hidden_dim 128 --max_epochs 100 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100 --temperature 0.1 --train_sample_ratio 1 --group_size 500 --seed 0 --replicate 1 --avg_thshd 95 --wg_thshd 85 --retest_dataset $d --retest_batch_size 128
done

# ERM Adapter (replicate 4), Target classes: Waterbirds
python main.py --dataset waterbirds --torch_seed --load_base_model clip_ViTL14 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layers 1 --hidden_dim 128 --max_epochs 100 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100 --temperature 0.1 --train_sample_ratio 1 --group_size 500 --seed 0 --replicate 4 --avg_thshd 95 --wg_thshd 85
for d in $RETEST_DATASETS; do
    python main_retest.py --dataset waterbirds --torch_seed --load_base_model clip_ViTL14 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layers 1 --hidden_dim 128 --max_epochs 100 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100 --temperature 0.1 --train_sample_ratio 1 --group_size 500 --seed 0 --replicate 4 --avg_thshd 95 --wg_thshd 85 --retest_dataset $d --retest_batch_size 128
done

# Contrastive Adapter (replicate 1), Target classes: CelebA
python main.py --dataset celebA --torch_seed --load_base_model clip_ViTL14 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layers 1 --hidden_dim 128 --max_epochs 50 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100 --temperature 0.1 --train_sample_ratio 0.1 --group_size 500 --seed 0 --replicate 1 --avg_thshd 93 --wg_thshd 42
for d in $RETEST_DATASETS; do
    python main_retest.py --dataset celebA --torch_seed --load_base_model clip_ViTL14 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layers 1 --hidden_dim 128 --max_epochs 50 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100 --temperature 0.1 --train_sample_ratio 0.1 --group_size 500 --seed 0 --replicate 1 --avg_thshd 93 --wg_thshd 42 --retest_dataset $d --retest_batch_size 128
done

# ERM Adapter (replicate 4), Target classes: CelebA
python main.py --dataset celebA --torch_seed --load_base_model clip_ViTL14 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layers 1 --hidden_dim 128 --max_epochs 50 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100 --temperature 0.1 --train_sample_ratio 0.1 --group_size 500 --seed 0 --replicate 4 --avg_thshd 93 --wg_thshd 42
for d in $RETEST_DATASETS; do
    python main_retest.py --dataset celebA --torch_seed --load_base_model clip_ViTL14 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layers 1 --hidden_dim 128 --max_epochs 50 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100 --temperature 0.1 --train_sample_ratio 0.1 --group_size 500 --seed 0 --replicate 4 --avg_thshd 93 --wg_thshd 42 --retest_dataset $d --retest_batch_size 128
done
