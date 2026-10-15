import os

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class Waterbirds(Dataset):

    def __init__(
        self,
        root_dir,
        target_name,
        confounder_names,
        split,
        augment_data=False,
        model_type=None,
        args=None,
        train_transform=None,
    ):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data
        self.split = split

        self.split_dict = {"train": 0, "val": 1, "test": 2}

        self.data_dir = os.path.join(
            self.root_dir, "_".join([self.target_name] + self.confounder_names)
        )

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )

        self.metadata_df = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))

        self.metadata_df = self.metadata_df[
            self.metadata_df["split"] == self.split_dict[self.split]
        ]

        self.y_array = self.metadata_df["y"].values
        self.n_classes = 2

        self.confounder_array = self.metadata_df["place"].values
        self.n_confounders = 1

        self.n_groups = pow(2, 2)
        self.group_array = (
            self.y_array * (self.n_groups / 2) + self.confounder_array
        ).astype("int")

        self.filename_array = self.metadata_df["img_filename"].values
        self.split_array = self.metadata_df["split"].values

        self.targets = torch.tensor(self.y_array)
        self.targets_all = {
            "target": np.array(self.y_array),
            "group_idx": np.array(self.group_array),
            "spurious": np.array(self.confounder_array),
            "sub_target": np.array(list(zip(self.y_array, self.confounder_array))),
        }
        self.group_labels = [
            "LANDBIRD on land",
            "LANDBIRD on water",
            "WATERBIRD on land",
            "WATERBIRD on water",
        ]
        self.class_names = ["LANDBIRD", "WATERBIRD"]

        self.train_transform = train_transform
        self.eval_transform = train_transform

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.targets[idx]

        img_filename = os.path.join(self.data_dir, self.filename_array[idx])
        img = Image.open(img_filename).convert("RGB")
        img = self.train_transform(img)
        x = img

        return (x, y, idx)

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups / self.n_classes)
        c = group_idx % (self.n_groups // self.n_classes)

        group_name = f"{self.target_name} = {int(y)}"
        bin_str = format(int(c), f"0{self.n_confounders}b")[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f", {attr_name} = {bin_str[attr_idx]}"
        return group_name


def load_waterbirds(args, train_shuffle=True, transform=None):
    train_set = Waterbirds(
        args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        split="train",
        model_type=args.arch,
        args=args,
        train_transform=transform,
    )
    train_loader = DataLoader(
        train_set, batch_size=128, shuffle=train_shuffle, num_workers=args.num_workers
    )

    val_set = Waterbirds(
        args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        split="val",
        model_type=args.arch,
        args=args,
        train_transform=transform,
    )
    val_loader = DataLoader(
        val_set, batch_size=128, shuffle=False, num_workers=args.num_workers
    )

    test_set = Waterbirds(
        args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        split="test",
        model_type=args.arch,
        args=args,
        train_transform=transform,
    )
    test_loader = DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=args.num_workers
    )
    args.num_classes = 2
    return (train_loader, val_loader, test_loader)


def load_dataloaders(args, train_shuffle=True, val_correlation=None, transform=None):
    return load_waterbirds(args, train_shuffle, transform)
