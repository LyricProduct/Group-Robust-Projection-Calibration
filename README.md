# Unleashing the Potential of Complementary Spaces in Group Robust Classification (Under Review)

This repository provides the PyTorch implementation of our paper  
**"Unleashing the Potential of Complementary Spaces in Group Robust Classification."**

Our code includes the implementations and evaluation pipelines used in the paper.

## Repository Structure

The repository is organized by method. We take one representative method from each category of CLIP-based group robust classification.  
Detailed setup instructions, implementation details, training procedures, and evaluation commands are provided in the `README.md` file of each subdirectory.

- `./Orth-Cali/`  
  Projection-based method (Orth-Cali) and our post-hoc calibration applied to Orth-Cali.

- `./PRISM/`  
  Linear probing method (PRISM).

- `./FairerCLIP/`  
  Non-linear probing method (FairerCLIP).

- `./ContrastiveAdapter/`  
  Adapter-based method (Contrastive Adapter).

Please refer to the `README.md` in each subdirectory for the exact environment setup and commands required to reproduce the corresponding experiments.

## Reproducing the Results

The experiments reported in the paper are reproduced separately within each method directory.  
Each subdirectory provides the commands for training or constructing the corresponding method and for evaluating both target group robustness and non-target generalization.

The main experimental settings, including dataset configurations, prompt templates, vocabulary settings, and hyperparameters, follow those described in the paper.

## Acknowledgements

Our implementations build on the official code of each method and on the following resources.

- Orth-Cali: [debias_vl](https://github.com/chingyaoc/debias_vl) (MIT License)
- PRISM: [PRISM](https://github.com/MahdiyarMM/PRISM)
- FairerCLIP: [FairerCLIP](https://github.com/human-analysis/FairerCLIP) (MIT License)
- Contrastive Adapter: the official code released with [Contrastive Adapters for Foundation Model Group Robustness](https://openreview.net/forum?id=T28aiF3-l-H) (NeurIPS 2022)
- CLIP: [OpenAI CLIP](https://github.com/openai/CLIP) (MIT License)
- Class names of the non-target datasets (`descriptors/`): [WaffleCLIP](https://github.com/ExplainableML/WaffleCLIP) (MIT License)
- ImageNet-V2: [ImageNetV2_pytorch](https://github.com/modestyachts/ImageNetV2_pytorch)
