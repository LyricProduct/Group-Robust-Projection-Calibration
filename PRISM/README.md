# Unleashing the Potential of Complementary Spaces in Group Robust Classification (Under Review)
We provide Pytorch implementation of our paper "Unleashing the Potential of Complementary Spaces in Group Robust Classification".
This directory contains the implementation and evaluation of PRISM (the learnable linear transformation variant).

## Setup

### 1. Environment
Please follow the steps below to build your environment.

```bash
# Create a conda environment
conda create -n prism python=3.11
conda activate prism

# Install torch and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip3 install torch torchvision

# Install dependencies
pip install -r requirements.txt

# Install CLIP
pip install git+https://github.com/openai/CLIP.git

# Install ImageNet-V2 package
pip install git+https://github.com/modestyachts/ImageNetV2_pytorch
```

### 2. Data Preparation
- Target datasets (Waterbirds, CelebA): They are loaded with the [WILDS](https://github.com/p-lambda/wilds) package and downloaded under `WATERBIRDS_DIR` / `CELEBA_DIR` when the code is run for the first time.
- Please set the paths for each dataset and the cache directories in ```configs/paths.py```.
- Non-target datasets: Download the datasets and set their directories in ```configs/paths.py```.
  - ImageNet (`IMAGENET_DIR`) is loaded with `torchvision.datasets.ImageNet` (validation split), so the directory must contain the official devkit and validation archives (or the extracted layout that torchvision expects).
  - Stanford Cars (`CARS_DIR`) must contain the test images under `cars_test/` and the test annotation file `cars_test_annos_withlabels.mat` renamed to `cars_test_annos_withlabels_renew`.
  - The other datasets (EuroSAT, Food-101, ImageNet-V2, Places365, Oxford-IIIT Pet, DTD, Flowers-102, FGVC-Aircraft) are loaded with `torchvision` / `imagenetv2_pytorch` and downloaded automatically when missing. CUB-200-2011 (`CUB_DIR`) has to be downloaded manually.
- The class names of the non-target datasets are provided in ```descriptors/```.
- The CLIP image embeddings of each non-target dataset are cached under `NON_TARGET_CLASS_CACHE_DIR` and reused in subsequent runs; delete the cache if you change the dataset or its directory.

### 3. Run
To reproduce the results of the paper, we provide examples of the execution commands below. All execution commands are documented in ```run_script.sh```.
The trained linear transformation is cached under `MODEL_CACHE_DIR` and reused when `--use_cache_P` is given, so the same command can be run for each non-target dataset without retraining.

- PRISM, Target classes: Waterbirds, Non-target classes: ImageNet
``` python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset waterbirds --CLIP_model ViT-L/14 --batch_size 64 --lr 0.1 --num_samples 500 --epochs 1 --seed 42 --init_weight random --reg_type None --reg_lambda 1e-3 --retest_dataset imagenet --use_cache_P
```

<br>

- PRISM, Target classes: CelebA, Non-target classes: ImageNet
``` python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset celeba --CLIP_model ViT-L/14 --batch_size 64 --lr 0.01 --num_samples 500 --epochs 1 --seed 42 --init_weight random --reg_type None --reg_lambda 1e-3 --retest_dataset imagenet --use_cache_P
```

The results are written to ```Result/<dataset>/<model>/result.csv```.

---

## Acknowledgements
- The PRISM implementation is based on the [official PRISM repository](https://github.com/MahdiyarMM/PRISM).
- CLIP is used through [OpenAI CLIP](https://github.com/openai/CLIP) (MIT License).
- The class names in `descriptors/` are taken from [WaffleCLIP](https://github.com/ExplainableML/WaffleCLIP) (MIT License).
- ImageNet-V2 is loaded with [ImageNetV2_pytorch](https://github.com/modestyachts/ImageNetV2_pytorch).
