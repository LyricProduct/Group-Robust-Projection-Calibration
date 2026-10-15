import copy
import importlib

import numpy as np
from torch.utils.data import DataLoader

from configs import paths


def initialize_data(args):
    args.resample_iid = False
    dataset_module = importlib.import_module(f"datasets.{args.dataset}")
    load_dataloaders = getattr(dataset_module, "load_dataloaders")

    args.results_dict = {
        "epoch": [],
        "dataset_ix": [],
        "train_loss": [],
        "train_avg_acc": [],
        "train_robust_acc": [],
        "val_loss": [],
        "val_avg_acc": [],
        "val_robust_acc": [],
        "test_loss": [],
        "test_avg_acc": [],
        "test_robust_acc": [],
        "best_loss_epoch": [],
        "best_acc_epoch": [],
        "best_robust_acc_epoch": [],
    }

    if args.dataset == "waterbirds":
        args.root_dir = paths.WATERBIRDS_DIR
        args.val_split = 0.2
        args.target_name = "waterbird_complete95"
        args.confounder_names = ["forest2water2"]
        args.augment_data = False
        args.train_classes = ["landbird", "waterbird"]
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.text_descriptions = ["a landbird", "a waterbird"]
        args.wilds_dataset = False

    elif args.dataset == "celebA":
        args.root_dir = paths.CELEBA_DIR
        args.target_name = "Blond_Hair"
        args.confounder_names = ["Male"]
        args.image_mean = np.mean([0.485, 0.456, 0.406])
        args.image_std = np.mean([0.229, 0.224, 0.225])
        args.augment_data = False
        args.image_path = "./images/celebA/"
        args.train_classes = ["nonblond", "blond"]
        args.val_split = 0.2
        args.text_descriptions = [
            "a celebrity with dark hair",
            "a celebrity with blond hair",
        ]
        args.wilds_dataset = False

    else:
        raise NotImplementedError

    args.task = args.dataset
    args.num_classes = len(args.train_classes)
    return load_dataloaders


def update_classification_prompts(args):
    if args.dataset == "waterbirds":
        args.text_descriptions = [
            f"This is a picture of a {c}." for c in args.train_classes
        ]
    elif args.dataset == "celebA":
        args.text_descriptions = [
            "A photo of a celebrity with dark hair.",
            "A photo of a celebrity with blond hair.",
        ]
    return args.text_descriptions


def get_resampled_set(dataset, resampled_set_indices, copy_dataset=False):
    resampled_set = copy.deepcopy(dataset) if copy_dataset else dataset
    try:
        resampled_set.y_array = resampled_set.y_array[resampled_set_indices]
        resampled_set.group_array = resampled_set.group_array[resampled_set_indices]
        resampled_set.split_array = resampled_set.split_array[resampled_set_indices]
        resampled_set.targets = resampled_set.y_array
        try:
            resampled_set.filename_array = resampled_set.filename_array[
                resampled_set_indices
            ]
        except:
            resampled_set.x_array = resampled_set.x_array[resampled_set_indices]
    except AttributeError as e:
        try:
            resampled_set.targets = resampled_set.targets[resampled_set_indices]
        except:
            resampled_set_indices = np.concatenate(resampled_set_indices)
            resampled_set.targets = resampled_set.targets[resampled_set_indices]
            print(f"Error: {e}")
    try:
        resampled_set.df = resampled_set.df.iloc[resampled_set_indices]
    except AttributeError:
        pass

    try:
        resampled_set.data = resampled_set.data[resampled_set_indices]
    except AttributeError:
        pass

    try:
        resampled_set.filename_array = resampled_set.filename_array[
            resampled_set_indices
        ]
    except:
        pass

    try:
        resampled_set.metadata_array = resampled_set.metadata_array[
            resampled_set_indices
        ]
    except:
        pass

    for target_type, target_val in resampled_set.targets_all.items():
        resampled_set.targets_all[target_type] = target_val[resampled_set_indices]

    return resampled_set


def get_sample_indices_by_class_by_pred(class_labels, predictions, num_classes):
    all_classes = range(num_classes)
    indices_by_class_by_pred = []

    for c in all_classes:
        indices_by_class = []
        for p in all_classes:
            indices = np.where(np.logical_and(class_labels == c, predictions == p))[0]
            indices_by_class.append(indices)
        indices_by_class_by_pred.append(indices_by_class)
    return indices_by_class_by_pred


def balance_groups(indices_by_group, resample="upsample", seed=42):
    np.random.seed(seed)
    resampled_indices = []
    sampling_indices = [i for i in indices_by_group if len(i) > 0]
    sizes = [len(i) for i in sampling_indices]

    if resample == "upsample":
        ix_size = np.argmax(sizes)
        resample = True
    else:
        ix_size = np.argmin(sizes)
        resample = False

    for ix, indices in enumerate(sampling_indices):
        resample_size = sizes[ix_size]
        if ix == ix_size:
            resampled_indices.append(indices)
        else:
            resampled_indices.append(
                np.random.choice(indices, size=resample_size, replace=resample)
            )
    return np.concatenate(resampled_indices)


def balance_preds_by_class(indices_by_class_by_pred, seed):
    all_resampled_indices = []
    for class_indices in indices_by_class_by_pred:
        resampled_class_indices = balance_groups(
            class_indices, resample="upsample", seed=seed
        )
        all_resampled_indices.append(resampled_class_indices)
    all_resampled_indices = np.concatenate(all_resampled_indices)
    return all_resampled_indices


def balance_dataset_class_group(predictions, initial_dataset, seed):
    np.random.seed(seed)

    targets = initial_dataset.targets_all["target"]
    try:
        predictions = predictions.numpy()
    except:
        pass

    indices_by_class = []
    for t in np.unique(targets):
        indices_by_class.append(np.where(targets == t)[0])
    balancing_indices = balance_groups(indices_by_class, seed=seed)
    _targets = targets[balancing_indices]
    _predictions = predictions[balancing_indices]

    indices_by_class_by_pred = get_sample_indices_by_class_by_pred(
        _targets, _predictions, num_classes=len(np.unique(targets))
    )

    _balancing_indices = balance_preds_by_class(indices_by_class_by_pred, seed)

    balancing_indices = balancing_indices[_balancing_indices]

    np.random.shuffle(balancing_indices)
    dataset_p = get_resampled_set(
        dataset=initial_dataset,
        resampled_set_indices=balancing_indices,
        copy_dataset=True,
    )
    return dataset_p


def resample_dataset_by_pred_by_class(
    predictions, initial_dataset, resample, group_size=None, seed=42
):
    np.random.seed(seed)
    targets = initial_dataset.targets_all["target"]
    all_sample_indices = []
    for p in np.unique(targets):
        p_ix = np.where(targets == p)[0]
        y_indices = [
            np.where(predictions[p_ix] == y)[0] for y in np.unique(predictions[p_ix])
        ]
        if resample == "upsample":
            max_ix = np.argmax([len(y_ix) for y_ix in y_indices])
            max_len = len(y_indices[max_ix])
            sample_indices = [p_ix[y_ix] for y_ix in copy.deepcopy(y_indices)]
            for ix, y_ix in enumerate(y_indices):
                if ix != max_ix:
                    sample_indices.append(
                        np.random.choice(
                            p_ix[y_ix], size=max_len - len(y_ix), replace=True, p=None
                        )
                    )
            p_ix = np.concatenate(sample_indices)
            np.random.shuffle(p_ix)
        elif resample == "subsample":
            min_ix = np.argmin([len(y_ix) for y_ix in y_indices])
            sample_indices = [p_ix[y_indices[min_ix]]]
            for ix, y_ix in enumerate(y_indices):
                if ix != min_ix:
                    sample_indices.append(
                        np.random.choice(
                            p_ix[y_ix],
                            size=len(y_indices[min_ix]),
                            replace=False,
                            p=None,
                        )
                    )
            p_ix = np.concatenate(sample_indices)

        elif resample == "group_size":
            assert group_size is not None
            sample_indices = []
            for ix, y_ix in enumerate(y_indices):
                replace = True if len(p_ix[y_ix]) < group_size else False
                sample_indices.append(
                    np.random.choice(
                        p_ix[y_ix], size=group_size, replace=replace, p=None
                    )
                )
            p_ix = np.concatenate(sample_indices)

        all_sample_indices.append(p_ix)

    p_ix = np.concatenate(all_sample_indices)
    np.random.shuffle(p_ix)
    dataset_p = get_resampled_set(
        dataset=initial_dataset, resampled_set_indices=p_ix, copy_dataset=True
    )
    return dataset_p


def get_resampled_loader_by_pred_by_class(
    predictions, initial_dataset, resample, seed, args, batch_size=None
):
    dataset = resample_dataset_by_pred_by_class(
        predictions, initial_dataset, resample, args.group_size, seed
    )
    batch_size = batch_size if batch_size is not None else args.bs_trn
    dataloader = DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=args.num_workers
    )
    return dataloader
