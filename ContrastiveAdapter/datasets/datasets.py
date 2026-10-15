import json
import os
import pathlib
import ssl

import torch
from PIL import Image
from scipy.io import loadmat
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.datasets import (
    DTD,
    EuroSAT,
    FGVCAircraft,
    Flowers102,
    Food101,
    ImageNet,
    OxfordIIITPet,
    Places365,
)
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2

from configs import paths


class CUBDataset(datasets.ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
        train=True,
        bboxes=False,
    ):
        img_root = os.path.join(root, "images")

        super(CUBDataset, self).__init__(
            root=img_root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        self.redefine_class_to_idx()

        self.transform_ = transform
        self.target_transform_ = target_transform
        self.train = train

        path_to_splits = os.path.join(root, "train_test_split.txt")
        indices_to_use = list()
        with open(path_to_splits, "r") as in_file:
            for line in in_file:
                idx, use_train = line.strip("\n").split(" ", 2)
                if bool(int(use_train)) == self.train:
                    indices_to_use.append(int(idx))

        path_to_index = os.path.join(root, "images.txt")
        filenames_to_use = set()
        with open(path_to_index, "r") as in_file:
            for line in in_file:
                idx, fn = line.strip("\n").split(" ", 2)
                if int(idx) in indices_to_use:
                    filenames_to_use.add(fn)

        img_paths_cut = {
            "/".join(img_path.rsplit("/", 2)[-2:]): idx
            for idx, (img_path, lb) in enumerate(self.imgs)
        }
        imgs_to_use = [self.imgs[img_paths_cut[fn]] for fn in filenames_to_use]

        _, targets_to_use = list(zip(*imgs_to_use))

        self.imgs = self.samples = imgs_to_use
        self.targets = targets_to_use

        if bboxes:
            path_to_bboxes = os.path.join(root, "bounding_boxes.txt")
            bounding_boxes = list()
            with open(path_to_bboxes, "r") as in_file:
                for line in in_file:
                    idx, x, y, w, h = map(
                        lambda x: float(x), line.strip("\n").split(" ")
                    )
                    if int(idx) in indices_to_use:
                        bounding_boxes.append((x, y, w, h))

            self.bboxes = bounding_boxes
        else:
            self.bboxes = None

    def __getitem__(self, index):
        sample, target = super(CUBDataset, self).__getitem__(index)

        if self.bboxes is not None:
            width, height = sample.width, sample.height
            x, y, w, h = self.bboxes[index]

            scale_resize = 500 / width
            scale_resize_crop = scale_resize * (375 / 500)

            x_rel = scale_resize_crop * x / 375
            y_rel = scale_resize_crop * y / 375
            w_rel = scale_resize_crop * w / 375
            h_rel = scale_resize_crop * h / 375

            target = torch.tensor([target, x_rel, y_rel, w_rel, h_rel])

        if self.transform_ is not None:
            sample = self.transform_(sample)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return sample, target

    def redefine_class_to_idx(self):
        adjusted_dict = {}
        for k, v in self.class_to_idx.items():
            k = k.split(".")[-1].replace("_", " ")
            split_key = k.split(" ")
            if len(split_key) > 2:
                k = "-".join(split_key[:-1]) + " " + split_key[-1]
            adjusted_dict[k] = v
        self.class_to_idx = adjusted_dict


class StanfordCars(torch.utils.data.Dataset):
    def __init__(self, root_path, transform=None):
        self.root_path = pathlib.Path(os.path.join(root_path, "cars_test"))
        file_path = os.path.join(root_path, "cars_test_annos_withlabels_renew")
        self.images = [
            os.path.join(self.root_path, file)
            for file in sorted(os.listdir(self.root_path))
        ]
        self.transform = transform

        annos = loadmat(file_path)
        annotations = annos["annotations"]
        labels = annotations["class"]
        self.labels = [int(label[0][0]) - 1 for label in labels[0]]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]


def _transform(n_px):
    return transforms.Compose(
        [
            transforms.Resize(n_px, interpolation=Image.BICUBIC),
            transforms.CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


RETEST_DATASETS = [
    "imagenet",
    "cub",
    "eurosat",
    "food101",
    "imagenetv2",
    "places365",
    "pets",
    "dtd",
    "flowers102",
    "fgvcaircraft",
    "cars",
]


def get_retest_classnames(retest_dataset):
    if retest_dataset in ("imagenet", "imagenetv2"):
        classnames = []
        with open("./descriptors/imagenet_classnames.txt", "r") as f:
            for line in f:
                classnames.append(line.strip().split(" ", 1)[1])
    else:
        with open(f"./descriptors/descriptors_{retest_dataset}.json", "r") as f:
            classnames = list(json.load(f).keys())
        if retest_dataset == "pets":
            classnames = sorted(classnames, key=lambda x: x.lower())
    return classnames


def get_retest_prompts(retest_dataset):
    return [
        (f"A photo of an {c}." if c[0] in "aeiou" else f"A photo of a {c}.")
        for c in get_retest_classnames(retest_dataset)
    ]


def load_retest_dataset(retest_dataset, transform):
    if retest_dataset == "imagenet":
        data_dir = pathlib.Path(paths.IMAGENET_DIR)
        dataset = ImageNet(data_dir, split="val", transform=transform)
    elif retest_dataset == "eurosat":
        ssl._create_default_https_context = ssl._create_unverified_context
        data_dir = pathlib.Path(paths.EUROSAT_DIR)
        dataset = EuroSAT(data_dir, transform=transform, download=True)
    elif retest_dataset == "cub":
        data_dir = pathlib.Path(paths.CUB_DIR)
        dataset = CUBDataset(data_dir, train=False, transform=transform)
    elif retest_dataset == "food101":
        data_dir = pathlib.Path(paths.FOOD101_DIR)
        dataset = Food101(data_dir, split="test", transform=transform, download=True)
    elif retest_dataset == "imagenetv2":
        data_dir = pathlib.Path(paths.IMAGENETV2_DIR)
        dataset = ImageNetV2(location=data_dir, transform=transform)
    elif retest_dataset == "places365":
        data_dir = pathlib.Path(paths.PLACES365_DIR)
        download = not os.path.exists(os.path.join(data_dir, "val_256"))
        dataset = Places365(
            data_dir,
            split="val",
            small=True,
            transform=transform,
            download=download,
        )
    elif retest_dataset == "pets":
        data_dir = pathlib.Path(paths.PETS_DIR)
        dataset = OxfordIIITPet(
            data_dir, split="test", transform=transform, download=True
        )
    elif retest_dataset == "dtd":
        data_dir = pathlib.Path(paths.DTD_DIR)
        dataset = DTD(data_dir, split="test", transform=transform, download=True)
    elif retest_dataset == "flowers102":
        data_dir = pathlib.Path(paths.FLOWERS102_DIR)
        dataset = Flowers102(data_dir, split="test", transform=transform, download=True)
    elif retest_dataset == "fgvcaircraft":
        data_dir = pathlib.Path(paths.FGVCAIRCRAFT_DIR)
        dataset = FGVCAircraft(
            data_dir,
            split="test",
            annotation_level="family",
            transform=transform,
            download=True,
        )
    elif retest_dataset == "cars":
        data_dir = pathlib.Path(paths.CARS_DIR)
        dataset = StanfordCars(data_dir, transform=transform)
    else:
        raise ValueError(f"Unknown non-target dataset: {retest_dataset}")
    return dataset
