import os

import numpy as np
import torch
from torch.utils.data import Dataset

from configs import paths


def to_pm1_onehot(values):
    return (
        torch.nn.functional.one_hot(torch.from_numpy(values.astype(int)), num_classes=2)
    ) * 2 - 1


def binarize_attribute(values):
    return (np.abs((values.astype(int) + 1) / 2)).astype(int)


class Clipfeature(Dataset):
    def __init__(self, split, cfg):
        features_dir = paths.FEATURES_DIR

        if cfg.dataset == "waterbirds":
            if split == "train":
                labels = np.loadtxt(
                    os.path.join(features_dir, "waterbirds_train.csv"),
                    delimiter=",",
                    dtype=str,
                )
                self.imfeat = torch.load(
                    os.path.join(
                        features_dir,
                        "d=waterbirds-s=train-m=%s.pt" % cfg.load_base_model,
                    ),
                ).to(cfg.device)
                self.labels_s = to_pm1_onehot(
                    np.loadtxt(
                        os.path.join(features_dir, "prediction-waterbirds-s-train.csv")
                    )
                )
                if cfg.nolabel == True:
                    labels_sample = np.loadtxt(
                        os.path.join(features_dir, "prediction-waterbirds-y-train.csv")
                    ).astype(int)
                    self.labels_y = to_pm1_onehot(labels_sample)
                    self.labels_y_gt = to_pm1_onehot(labels[:, 2])
                else:
                    self.labels_y = to_pm1_onehot(labels[:, 2])
                    self.labels_y_gt = self.labels_y
                    labels_sample = labels[:, 2].astype(int)

                data_y0 = np.where(labels_sample == 1)[0]
                data_y1 = np.where(labels_sample == 0)[0]
                data_y0 = np.random.permutation(data_y0)[
                    : int(len(data_y0) * cfg.sample_ratio1)
                ]
                data_y1 = np.random.permutation(data_y1)[
                    : int(len(data_y1) * cfg.sample_ratio2)
                ]
                data = np.hstack((data_y0, data_y1))
                data.sort()
                self.imfeat = self.imfeat[data]
                self.labels_s = self.labels_s[data]
                self.labels_y = self.labels_y[data]
                self.labels_y_gt = self.labels_y_gt[data]

            elif split == "test":
                self.imfeat = torch.load(
                    os.path.join(
                        features_dir,
                        "d=waterbirds-s=test-m=%s.pt" % cfg.load_base_model,
                    ),
                ).to(cfg.device)
                labels = np.loadtxt(
                    os.path.join(features_dir, "waterbirds_test.csv"),
                    delimiter=",",
                    dtype=str,
                )
                self.labels_s = to_pm1_onehot(labels[:, 4])
                self.labels_y = to_pm1_onehot(labels[:, 2])
                self.labels_y_gt = self.labels_y

        elif cfg.dataset == "celebA":
            if split == "train":
                labels = np.loadtxt(
                    os.path.join(features_dir, "celeba_train.csv"),
                    delimiter=",",
                    dtype=str,
                )
                self.imfeat = torch.load(
                    os.path.join(
                        features_dir, "d=celebA-s=train-m=%s.pt" % cfg.load_base_model
                    ),
                ).to(cfg.device)
                self.labels_s = to_pm1_onehot(
                    np.loadtxt(
                        os.path.join(features_dir, "prediction-celeba-s-train.csv")
                    )
                )

                if cfg.nolabel == True:
                    labels_sample = np.loadtxt(
                        os.path.join(features_dir, "prediction-celeba-y-train.csv")
                    ).astype(int)
                    self.labels_y = to_pm1_onehot(labels_sample)
                    self.labels_y_gt = torch.nn.functional.one_hot(
                        torch.from_numpy(binarize_attribute(labels[:, 10])),
                        num_classes=2,
                    )
                    self.labels_y_gt[self.labels_y_gt == 0] = -1
                else:
                    self.labels_y = torch.nn.functional.one_hot(
                        torch.from_numpy(binarize_attribute(labels[:, 10])),
                        num_classes=2,
                    )
                    self.labels_y[self.labels_y == 0] = -1
                    self.labels_y_gt = self.labels_y
                    labels_sample = (labels[:, 10].astype(int) + 1) / 2

                data_y0 = np.where(labels_sample == 1)[0]
                data_y1 = np.where(labels_sample == 0)[0]
                balance_sample = min(len(data_y0), len(data_y1))
                data_y0 = np.random.permutation(data_y0)[
                    : int(balance_sample * cfg.sample_ratio1)
                ]
                data_y1 = np.random.permutation(data_y1)[
                    : int(balance_sample * cfg.sample_ratio2)
                ]
                data = np.hstack((data_y0, data_y1))
                data.sort()
                self.imfeat = self.imfeat[data]
                self.labels_s = self.labels_s[data]
                self.labels_y = self.labels_y[data]
                self.labels_y_gt = self.labels_y_gt[data]

            elif split == "test":
                self.imfeat = torch.load(
                    os.path.join(
                        features_dir, "d=celebA-s=test-m=%s.pt" % cfg.load_base_model
                    ),
                ).to(cfg.device)
                labels = np.loadtxt(
                    os.path.join(features_dir, "celeba_test.csv"),
                    delimiter=",",
                    dtype=str,
                )
                self.labels_s = torch.nn.functional.one_hot(
                    torch.from_numpy(binarize_attribute(labels[:, 21])), num_classes=2
                )
                self.labels_y = torch.nn.functional.one_hot(
                    torch.from_numpy(binarize_attribute(labels[:, 10])), num_classes=2
                )
                self.labels_y[self.labels_y == 0] = -1
                self.labels_s[self.labels_s == 0] = -1
                self.labels_y_gt = self.labels_y

        else:
            raise RuntimeError("No matched dataset (waterbirds / celebA)")

    def __len__(self):
        return len(self.imfeat)

    def __getitem__(self, idx):
        return (
            self.imfeat[idx],
            self.labels_y[idx],
            self.labels_s[idx],
            self.labels_y_gt[idx],
        )
