import importlib

import numpy as np

from configs import paths


def initialize_data(args):
    dataset = args.dataset.split("_")[0]
    dataset_module = importlib.import_module(f"datasets.{dataset}")
    load_dataloaders = getattr(dataset_module, "load_dataloaders")

    if args.dataset == "waterbirds":

        args.root_dir = paths.WATERBIRDS_DIR
        args.target_name = "waterbird_complete95"
        args.confounder_names = ["forest2water2"]
        args.augment_data = False
        args.train_classes = ["landbird", "waterbird"]

        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.text_descriptions = ["a landbird", "a waterbird"]

    elif "celebA" in args.dataset:
        args.root_dir = paths.CELEBA_DIR

        args.target_name = "Blond_Hair"
        args.confounder_names = ["Male"]
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.train_classes = ["nonblond", "blond"]
        args.text_descriptions = [
            "a celebrity with dark hair",
            "a celebrity with blond hair",
        ]

    else:
        raise NotImplementedError

    args.task = args.dataset
    try:
        args.num_classes = len(args.train_classes)
    except:
        pass
    return load_dataloaders
