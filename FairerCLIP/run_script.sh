#!/bin/bash

RETEST_DATASETS="imagenet cub eurosat food101 imagenetv2 places365 pets dtd flowers102 fgvcaircraft cars"

# FairerCLIP (w/ labels), Target classes: Waterbirds
for d in $RETEST_DATASETS; do
    uv run python train_retest.py --config-file configs/debias_waterbird.yaml --seed 0 --retest_dataset $d
done

# FairerCLIP (w/ labels), Target classes: CelebA
for d in $RETEST_DATASETS; do
    uv run python train_retest.py --config-file configs/debias_celeba.yaml --seed 0 --retest_dataset $d
done

# FairerCLIP (w/o labels), Target classes: Waterbirds
for d in $RETEST_DATASETS; do
    uv run python train_retest.py --config-file configs/debias_waterbird.yaml --opts nolabel True --seed 0 --retest_dataset $d
done

# FairerCLIP (w/o labels), Target classes: CelebA
for d in $RETEST_DATASETS; do
    uv run python train_retest.py --config-file configs/debias_celeba.yaml --opts nolabel True --seed 0 --retest_dataset $d
done
