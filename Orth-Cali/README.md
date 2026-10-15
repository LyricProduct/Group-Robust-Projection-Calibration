# Unleashing the Potential of Complementary Spaces in Group Robust Classification (Under Review)
We provide Pytorch implementation of our paper "Unleashing the Potential of Complementary Spaces in Group Robust Classification".

## Setup

### 1. Environment
Please follow the steps below to build your environment.

```bash
# Create a conda environment 
conda create -n orthcali python=3.11
conda activate orthcali

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

- Zero-shot CLIP, Target classes: Waterbirds, Non-target classes: ImageNet
``` python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset waterbirds --load_base_model clip_ViTL14 --retest_dataset imagenet
```

<br>

- Orth-Cali, Target classes: Waterbirds, Non-target classes: ImageNet
``` python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset waterbirds --load_base_model clip_ViTL14 --beta 0 --debias --retest_dataset imagenet
```

<br>

- Orth-Cali + Ours (ImageNet), Target classes: Waterbirds, Non-target classes: ImageNet
``` python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 500 --beta 2.0 --gamma 1.0 --debias --retest_dataset imagenet --T_mode IN_syns
```

<br>

- Orth-Cali + Ours (Wiki), Target classes: Waterbirds, Non-target classes: ImageNet
``` python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset imagenet --T_mode wiki --T_size 2500
```

<br>

- Orth-Cali + Ours (Google), Target classes: Waterbirds, Non-target classes: ImageNet
``` python
CUDA_VISIBLE_DEVICES=0 python main.py --dataset waterbirds --load_base_model clip_ViTL14 --alpha 100 --beta 10.0 --gamma 10.0 --debias --retest_dataset imagenet --T_mode google --T_size 2500
```

---

## Acknowledgements
- The Orth-Cali implementation is based on [debias_vl](https://github.com/chingyaoc/debias_vl) (MIT License).
- `clip/` is adapted from [OpenAI CLIP](https://github.com/openai/CLIP) (MIT License, see `clip/LICENSE`).
- The class names in `descriptors/` are taken from [WaffleCLIP](https://github.com/ExplainableML/WaffleCLIP) (MIT License).
- ImageNet-V2 is loaded with [ImageNetV2_pytorch](https://github.com/modestyachts/ImageNetV2_pytorch).
