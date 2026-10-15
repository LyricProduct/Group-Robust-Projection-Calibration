# Unleashing the Potential of Complementary Spaces in Group Robust Classification (Under Review)
We provide Pytorch implementation of our paper "Unleashing the Potential of Complementary Spaces in Group Robust Classification".
This directory contains the implementation and evaluation of Contrastive Adapter on both target group robustness and non-target generalization.

## Setup

### 1. Environment
Please follow the steps below to build your environment.

```bash
# Create a conda environment
conda create -n contrastive_adapter python=3.11
conda activate contrastive_adapter

# Install torch and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip3 install torch torchvision

# Install dependencies
pip install -r requirements.txt

# Install ImageNet-V2 package
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
```

### 2. Data Preparation
- Waterbirds: Download the dataset from [here](https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz). Unzipping this should result in a folder `waterbird_complete95_forest2water2`, which should be placed under `WATERBIRDS_DIR`.
- CelebA: Download the dataset files from this [Kaggle link](https://www.kaggle.com/jessicali9530/celeba-dataset). Then place the files under `CELEBA_DIR` such that we have the following structure:
```
# In CELEBA_DIR:
|-- list_attr_celeba.csv
|-- list_eval_partition.csv
|-- img_align_celeba/
    |-- image1.png
    |-- ...
    |-- imageN.png
```
- Please set the paths for each dataset and the cache directories in ```configs/paths.py```.
- Non-target datasets: Download the datasets and set their directories in ```configs/paths.py```.
  - ImageNet (`IMAGENET_DIR`) is loaded with `torchvision.datasets.ImageNet` (validation split), so the directory must contain the official devkit and validation archives (or the extracted layout that torchvision expects).
  - Stanford Cars (`CARS_DIR`) must contain the test images under `cars_test/` and the test annotation file `cars_test_annos_withlabels.mat` renamed to `cars_test_annos_withlabels_renew`.
  - The other datasets (EuroSAT, Food-101, ImageNet-V2, Places365, Oxford-IIIT Pet, DTD, Flowers-102, FGVC-Aircraft) are loaded with `torchvision` / `imagenetv2_pytorch` and downloaded automatically when missing. CUB-200-2011 (`CUB_DIR`) has to be downloaded manually.
- The class names of the non-target datasets are provided in ```descriptors/```.
- The CLIP image embeddings of each non-target dataset are cached under `NON_TARGET_CLASS_CACHE_DIR` and reused in subsequent runs; delete the cache if you change the dataset or its directory.

### 3. Run
To reproduce the results of the paper, we provide examples of the execution commands below. All execution commands are documented in ```run_script.sh```.
```main.py``` trains the adapter and evaluates it on the target dataset. The checkpoints with the best validation loss / average accuracy / worst-group accuracy are saved under ```models/<dataset>/```, and the per-epoch metrics are written to ```results/<dataset>/```.
```main_retest.py``` loads the trained adapter and additionally evaluates it on the non-target dataset given by `--retest_dataset`.
`--replicate 1` trains the contrastive adapter (contrastive epochs alternate with cross-entropy epochs on a group-balanced subset), and `--replicate 4` trains the adapter with the standard cross-entropy loss (ERM).
The CLIP embeddings of the target dataset and the adapter embeddings are cached under `TARGET_CLASS_CACHE_DIR` / `ADAPTER_CACHE_DIR`.

- Contrastive Adapter, Target classes: Waterbirds
``` python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset waterbirds --torch_seed --load_base_model clip_ViTL14 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layers 1 --hidden_dim 128 --max_epochs 100 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100 --temperature 0.1 --train_sample_ratio 1 --group_size 500 --seed 0 --replicate 1 --avg_thshd 95 --wg_thshd 85
```

<br>

- Contrastive Adapter, Target classes: Waterbirds, Non-target classes: ImageNet
``` python
CUDA_VISIBLE_DEVICES=0 python main_retest.py --dataset waterbirds --torch_seed --load_base_model clip_ViTL14 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layers 1 --hidden_dim 128 --max_epochs 100 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100 --temperature 0.1 --train_sample_ratio 1 --group_size 500 --seed 0 --replicate 1 --avg_thshd 95 --wg_thshd 85 --retest_dataset imagenet --retest_batch_size 128
```

<br>

- Contrastive Adapter, Target classes: CelebA
``` python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset celebA --torch_seed --load_base_model clip_ViTL14 --train_method adapter_mt --adapter_head_batch_norm --num_encoder_layers 1 --hidden_dim 128 --max_epochs 50 --bs_trn 128 --bs_val 128 --optimizer sgd --lr 1e-3 --weight_decay 5e-5 --momentum 0.9 --num_positives 2048 --num_negatives 2048 --num_neighbors_for_negatives 4096 --classification_temperature 100 --temperature 0.1 --train_sample_ratio 0.1 --group_size 500 --seed 0 --replicate 1 --avg_thshd 93 --wg_thshd 42
```

The results of ```main_retest.py``` are written to ```Result/<dataset>/result_r=<replicate>.csv```.

---

## Acknowledgements
- The Contrastive Adapter implementation is based on the official code released with the paper [Contrastive Adapters for Foundation Model Group Robustness](https://openreview.net/forum?id=T28aiF3-l-H) (NeurIPS 2022).
- `clip/` is adapted from [OpenAI CLIP](https://github.com/openai/CLIP) (MIT License, see `clip/LICENSE`).
- The class names in `descriptors/` are taken from [WaffleCLIP](https://github.com/ExplainableML/WaffleCLIP) (MIT License).
- ImageNet-V2 is loaded with [ImageNetV2_pytorch](https://github.com/modestyachts/ImageNetV2_pytorch).
