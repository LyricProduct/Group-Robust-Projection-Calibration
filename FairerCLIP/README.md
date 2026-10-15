# Unleashing the Potential of Complementary Spaces in Group Robust Classification (Under Review)
We provide Pytorch implementation of our paper "Unleashing the Potential of Complementary Spaces in Group Robust Classification".
This directory contains the implementation and evaluation of FairerCLIP on both target group robustness and non-target generalization.

## Setup

### 1. Environment
The environment is managed with [uv](https://docs.astral.sh/uv/). Please follow the steps below to build your environment.

```bash
# Create the virtual environment and install dependencies (torch / torchvision for CUDA 12.8)
# Please edit pyproject.toml if you need a different cuda version
uv sync
```

### 2. Data Preparation
- Target datasets (Waterbirds, CelebA): FairerCLIP is trained on pre-extracted CLIP features. Please download the features and labels from the [official FairerCLIP repository](https://github.com/human-analysis/FairerCLIP) ([Google Drive](https://drive.google.com/file/d/12rp60K_ifPicHxPHnXlD5v77HeMX8qaK/view?usp=sharing)) and place them in `FEATURES_DIR`.
  `FEATURES_DIR` must contain, for each dataset, the image features `d=<waterbirds|celebA>-s=<train|test>-m=clip_ViTL14.pt`, the label files `<waterbirds|celeba>_<train|test>.csv`, and the zero-shot predictions on the training split `prediction-<waterbirds|celeba>-<s|y>-train.csv`.
  Alternatively, you can extract them from the raw images with ```prepare.py``` after setting `WATERBIRDS_DIR` / `CELEBA_DIR` (Waterbirds: `WATERBIRDS_DIR/waterbird_complete95_forest2water2/`, CelebA: the Kaggle layout with `list_attr_celeba.csv`, `list_eval_partition.csv` and `img_align_celeba/`).
  ```bash
  uv run python prepare.py --config-file configs/debias_waterbird.yaml
  uv run python prepare.py --config-file configs/debias_celeba.yaml
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
```train.py``` only evaluates the target dataset and prints the result, and ```train_retest.py``` additionally evaluates the non-target dataset given by `--retest_dataset`.
The FairerCLIP embeddings of the non-target dataset are cached under `NON_TARGET_KERNEL_CACHE_DIR` (keyed by the datasets, the label setting and the seed); delete the cache if you change the hyperparameters via `--opts`.

- FairerCLIP (w/ labels), Target classes: Waterbirds
``` python
CUDA_VISIBLE_DEVICES=0 uv run python train.py --config-file configs/debias_waterbird.yaml --seed 0
```

<br>

- FairerCLIP (w/o labels), Target classes: Waterbirds
``` python
CUDA_VISIBLE_DEVICES=0 uv run python train.py --config-file configs/debias_waterbird.yaml --opts nolabel True --seed 0
```

<br>

- FairerCLIP (w/ labels), Target classes: Waterbirds, Non-target classes: ImageNet
``` python
CUDA_VISIBLE_DEVICES=0 uv run python train_retest.py --config-file configs/debias_waterbird.yaml --seed 0 --retest_dataset imagenet
```

<br>

- FairerCLIP (w/o labels), Target classes: CelebA, Non-target classes: ImageNet
``` python
CUDA_VISIBLE_DEVICES=0 uv run python train_retest.py --config-file configs/debias_celeba.yaml --opts nolabel True --seed 0 --retest_dataset imagenet
```

The results are written to ```Result/<dataset>/result_<wlabels|wolabels>.csv```.

---

## Acknowledgements
- The FairerCLIP implementation is based on the [official FairerCLIP repository](https://github.com/human-analysis/FairerCLIP) (MIT License).
- `clip/` is adapted from [OpenAI CLIP](https://github.com/openai/CLIP) (MIT License, see `clip/LICENSE`).
- The class names in `descriptors/` are taken from [WaffleCLIP](https://github.com/ExplainableML/WaffleCLIP) (MIT License).
- ImageNet-V2 is loaded with [ImageNetV2_pytorch](https://github.com/modestyachts/ImageNetV2_pytorch).
