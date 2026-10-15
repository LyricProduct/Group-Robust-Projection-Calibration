#!/bin/bash

RETEST_DATASETS="imagenet cub eurosat food101 imagenetv2 places365 pets dtd flowers102 fgvcaircraft cars"

# PRISM, Target classes: Waterbirds
for d in $RETEST_DATASETS; do
    python main.py --dataset waterbirds --CLIP_model ViT-L/14 --batch_size 64 --lr 0.1 --num_samples 500 --epochs 1 --seed 42 --init_weight random --reg_type None --reg_lambda 1e-3 --retest_dataset $d --use_cache_P
done

# PRISM, Target classes: CelebA
for d in $RETEST_DATASETS; do
    python main.py --dataset celeba --CLIP_model ViT-L/14 --batch_size 64 --lr 0.01 --num_samples 500 --epochs 1 --seed 42 --init_weight random --reg_type None --reg_lambda 1e-3 --retest_dataset $d --use_cache_P
done
