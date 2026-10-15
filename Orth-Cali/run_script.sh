#!/bin/bash

# Zero-shot CLIP, Target classes: Waterbirds
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --retest_dataset imagenet
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --retest_dataset cub
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --retest_dataset eurosat
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --retest_dataset food101
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --retest_dataset imagenetv2
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --retest_dataset places365
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --retest_dataset pets
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --retest_dataset dtd
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --retest_dataset flowers102
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --retest_dataset fgvcaircraft
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --retest_dataset cars

# Zero-shot CLIP, Target classes: CelebA
python main.py --dataset celebA --load_base_model clip_ViTL14 --retest_dataset imagenet
python main.py --dataset celebA --load_base_model clip_ViTL14 --retest_dataset cub
python main.py --dataset celebA --load_base_model clip_ViTL14 --retest_dataset eurosat
python main.py --dataset celebA --load_base_model clip_ViTL14 --retest_dataset food101
python main.py --dataset celebA --load_base_model clip_ViTL14 --retest_dataset imagenetv2
python main.py --dataset celebA --load_base_model clip_ViTL14 --retest_dataset places365
python main.py --dataset celebA --load_base_model clip_ViTL14 --retest_dataset pets
python main.py --dataset celebA --load_base_model clip_ViTL14 --retest_dataset dtd
python main.py --dataset celebA --load_base_model clip_ViTL14 --retest_dataset flowers102
python main.py --dataset celebA --load_base_model clip_ViTL14 --retest_dataset fgvcaircraft
python main.py --dataset celebA --load_base_model clip_ViTL14 --retest_dataset cars


# Orth-Cali, Target classes: Waterbirds
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset imagenet
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset cub
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset eurosat
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset food101
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset imagenetv2
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset places365
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset pets
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset dtd
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset flowers102
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset fgvcaircraft
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset cars

# Orth-Cali, Target classes: CelebA
python main.py --dataset celebA --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset imagenet
python main.py --dataset celebA --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset cub
python main.py --dataset celebA --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset eurosat
python main.py --dataset celebA --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset food101
python main.py --dataset celebA --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset imagenetv2
python main.py --dataset celebA --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset places365
python main.py --dataset celebA --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset pets
python main.py --dataset celebA --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset dtd
python main.py --dataset celebA --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset flowers102
python main.py --dataset celebA --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset fgvcaircraft
python main.py --dataset celebA --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset cars


# Orth-Cali + Ours (ImageNet), Target classes: Waterbirds
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 500 --beta 2.0 --gamma 1.0 --debias --retest_dataset imagenet --T_mode IN_syns
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 500 --beta 2.0 --gamma 1.0 --debias --retest_dataset cub --T_mode IN_syns
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 500 --beta 2.0 --gamma 1.0 --debias --retest_dataset eurosat --T_mode IN_syns
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 500 --beta 2.0 --gamma 1.0 --debias --retest_dataset food101 --T_mode IN_syns
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 500 --beta 2.0 --gamma 1.0 --debias --retest_dataset imagenetv2 --T_mode IN_syns
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 500 --beta 2.0 --gamma 1.0 --debias --retest_dataset places365 --T_mode IN_syns
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 500 --beta 2.0 --gamma 1.0 --debias --retest_dataset pets --T_mode IN_syns
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 500 --beta 2.0 --gamma 1.0 --debias --retest_dataset dtd --T_mode IN_syns
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 500 --beta 2.0 --gamma 1.0 --debias --retest_dataset flowers102 --T_mode IN_syns
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 500 --beta 2.0 --gamma 1.0 --debias --retest_dataset fgvcaircraft --T_mode IN_syns
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 500 --beta 2.0 --gamma 1.0 --debias --retest_dataset cars --T_mode IN_syns

# Orth-Cali + Ours (ImageNet), Target classes: CelebA
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 2.0 --gamma 0.5 --debias --retest_dataset imagenet --T_mode IN_syns
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 2.0 --gamma 0.5 --debias --retest_dataset cub --T_mode IN_syns
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 2.0 --gamma 0.5 --debias --retest_dataset eurosat --T_mode IN_syns
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 2.0 --gamma 0.5 --debias --retest_dataset food101 --T_mode IN_syns
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 2.0 --gamma 0.5 --debias --retest_dataset imagenetv2 --T_mode IN_syns
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 2.0 --gamma 0.5 --debias --retest_dataset places365 --T_mode IN_syns
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 2.0 --gamma 0.5 --debias --retest_dataset pets --T_mode IN_syns
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 2.0 --gamma 0.5 --debias --retest_dataset dtd --T_mode IN_syns
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 2.0 --gamma 0.5 --debias --retest_dataset flowers102 --T_mode IN_syns
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 2.0 --gamma 0.5 --debias --retest_dataset fgvcaircraft --T_mode IN_syns
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 2.0 --gamma 0.5 --debias --retest_dataset cars --T_mode IN_syns


# Orth-Cali + Ours (Wiki), Target classes: Waterbirds
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset imagenet --T_mode wiki --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset cub --T_mode wiki --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset eurosat --T_mode wiki --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset food101 --T_mode wiki --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset imagenetv2 --T_mode wiki --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset places365 --T_mode wiki --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset pets --T_mode wiki --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset dtd --T_mode wiki --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset flowers102 --T_mode wiki --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset fgvcaircraft --T_mode wiki --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset cars --T_mode wiki --T_size 2500

# Orth-Cali + Ours (Wiki), Target classes: CelebA
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset imagenet --T_mode wiki --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset cub --T_mode wiki --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset eurosat --T_mode wiki --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset food101 --T_mode wiki --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset imagenetv2 --T_mode wiki --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset places365 --T_mode wiki --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset pets --T_mode wiki --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset dtd --T_mode wiki --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset flowers102 --T_mode wiki --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset fgvcaircraft --T_mode wiki --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset cars --T_mode wiki --T_size 2500


# Orth-Cali + Ours (Google), Target classes: Waterbirds
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset imagenet --T_mode google --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset cub --T_mode google --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset eurosat --T_mode google --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset food101 --T_mode google --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset imagenetv2 --T_mode google --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset places365 --T_mode google --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset pets --T_mode google --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset dtd --T_mode google --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset flowers102 --T_mode google --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset fgvcaircraft --T_mode google --T_size 2500
python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset cars --T_mode google --T_size 2500

# Orth-Cali + Ours (Google), Target classes: CelebA
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset imagenet --T_mode google --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset cub --T_mode google --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset eurosat --T_mode google --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset food101 --T_mode google --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset imagenetv2 --T_mode google --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset places365 --T_mode google --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset pets --T_mode google --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset dtd --T_mode google --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset flowers102 --T_mode google --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset fgvcaircraft --T_mode google --T_size 2500
python main.py --dataset celebA --load_base_model clip_ViTL14 --alpha 1000 --beta 10.0 --gamma 0.1 --debias --retest_dataset cars --T_mode google --T_size 2500
