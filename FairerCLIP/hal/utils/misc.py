import torch


def mean_center(features, dim):
    features = features.float()
    return features - torch.mean(features, dim=dim)
