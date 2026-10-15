import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CelebA(Dataset):
    _normalization_stats = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}

    def __init__(
        self,
        root_dir,
        target_name="Blond_Hair",
        confounder_names=["Male"],
        split="train",
        augment_data=False,
        model_type=None,
        train_transform=None,
    ):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names

        confounder_names = self.confounder_names[0]
        self.model_type = model_type
        if "_pt" in model_type:
            self.model_type = model_type[:-3]
        self.augment_data = augment_data
        self.split = split

        self.split_dict = {"train": 0, "val": 1, "test": 2}

        self.data_dir = self.root_dir

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )

        self.metadata_df = pd.read_csv(
            os.path.join(self.data_dir, "list_attr_celeba.csv")
        )
        self.split_df = pd.read_csv(
            os.path.join(self.data_dir, "list_eval_partition.csv")
        )

        self.metadata_df["partition"] = self.split_df["partition"]
        self.metadata_df = self.metadata_df[
            self.split_df["partition"] == self.split_dict[self.split]
        ]

        self.y_array = self.metadata_df[self.target_name].values
        self.confounder_array = self.metadata_df[confounder_names].values
        self.y_array = self.y_array.copy()
        self.y_array[self.y_array == -1] = 0
        self.confounder_array = self.confounder_array.copy()
        self.confounder_array[self.confounder_array == -1] = 0
        self.n_classes = len(np.unique(self.y_array))
        self.n_confounders = len(confounder_names)

        self.metadata_df["sub_target"] = (
            self.metadata_df[self.target_name].astype(str)
            + "_"
            + self.metadata_df[confounder_names].astype(str)
        )

        attributes = [self.target_name, confounder_names]
        self.df_groups = (
            self.metadata_df[attributes].groupby(attributes).size().reset_index()
        )
        self.df_groups["group_id"] = (
            self.df_groups[self.target_name].astype(str)
            + "_"
            + self.df_groups[confounder_names].astype(str)
        )
        self.subclass_map = (
            self.df_groups["group_id"]
            .reset_index()
            .set_index("group_id")
            .to_dict()["index"]
        )
        self.group_array = self.metadata_df["sub_target"].map(self.subclass_map).values
        groups, group_counts = np.unique(self.group_array, return_counts=True)
        self.n_groups = len(groups)

        self.filename_array = self.metadata_df["image_id"].values
        self.split_array = self.metadata_df["partition"].values

        self.targets = torch.tensor(self.y_array)
        self.targets_all = {
            "target": np.array(self.y_array),
            "group_idx": np.array(self.group_array),
            "spurious": np.array(self.confounder_array),
            "sub_target": np.array(list(zip(self.y_array, self.confounder_array))),
        }
        self.group_labels = [self.group_str(i) for i in range(self.n_groups)]
        self.features_mat = None

        if train_transform is not None:
            self.train_transform = train_transform
        else:
            self.train_transform = get_transform_celeba(self.model_type, train=True)

        if train_transform is not None:
            self.eval_transform = train_transform
        else:
            self.eval_transform = get_transform_celeba(self.model_type, train=True)

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.targets[idx]
        img_filename = os.path.join(
            self.data_dir, "img_align_celeba", self.filename_array[idx]
        )
        img = Image.open(img_filename)

        if self.split_array[idx] == self.split_dict["train"] and self.train_transform:
            img = self.train_transform(img)
        elif (
            self.split_array[idx] in [self.split_dict["val"], self.split_dict["test"]]
            and self.eval_transform
        ):
            img = self.eval_transform(img)
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


def get_transform_celeba(model_type, train):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    transform = transforms.Compose(
        [
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=CelebA._normalization_stats["mean"],
                std=CelebA._normalization_stats["std"],
            ),
        ]
    )
    return transform


def load_celeba(args, train_shuffle=True, transform=None):
    train_set = CelebA(
        args.root_dir, split="train", model_type=args.arch, train_transform=transform
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs_trn,
        shuffle=train_shuffle,
        num_workers=args.num_workers,
    )

    val_set = CelebA(
        args.root_dir, split="val", model_type=args.arch, train_transform=transform
    )
    val_loader = DataLoader(
        val_set, batch_size=args.bs_val, shuffle=False, num_workers=args.num_workers
    )

    test_set = CelebA(
        args.root_dir, split="test", model_type=args.arch, train_transform=transform
    )
    test_loader = DataLoader(
        test_set, batch_size=args.bs_val, shuffle=False, num_workers=args.num_workers
    )
    args.num_classes = 2

    return (train_loader, val_loader, test_loader)


def load_dataloaders(args, train_shuffle=True, val_correlation=None, transform=None):
    return load_celeba(args, train_shuffle, transform)
