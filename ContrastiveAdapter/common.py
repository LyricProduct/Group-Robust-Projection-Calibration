import argparse
import os

import numpy as np
import torch

from clip import clip
from configs import paths
from datasets.datasets import RETEST_DATASETS
from datasets import initialize_data, update_classification_prompts
from datasets.embeddings import get_embedding_loaders_by_split
from evaluate import (
    evaluate_dataset_prediction,
    get_adapter_embeddings,
    get_adapter_predictions,
)
from network import get_adapter, load_base_model
from network.clip import evaluate_clip
from utils import initialize_experiment


def get_args():
    parser = argparse.ArgumentParser(description="CLIP Contrastive Embedding Adapter")

    parser.add_argument("--dataset", type=str, default="waterbirds")
    parser.add_argument("--num_workers", default=2, type=int)

    parser.add_argument("--load_base_model", type=str, default="clip_ViTL14")

    parser.add_argument("--train_method", default="adapter_mt", choices=["adapter_mt"])
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--adapter_head_batch_norm", default=False, action="store_true")
    parser.add_argument("--residual_connection", default=False, action="store_true")
    parser.add_argument(
        "--activation", type=str, default="relu", choices=["relu", "gelu"]
    )
    parser.add_argument("--classification_temperature", type=float, default=100)
    parser.add_argument("--num_encoder_layers", default=1, type=int)
    parser.add_argument("--zero_shot", default=False, action="store_true")

    parser.add_argument(
        "--contrastive_method", type=str, default="supcon", choices=["simclr", "supcon"]
    )
    parser.add_argument("--num_positives", type=int, default=2048)
    parser.add_argument("--num_negatives", type=int, default=2048)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument(
        "--balance_positive_classes", default=False, action="store_true"
    )
    parser.add_argument("--balance_anchor_classes", default=False, action="store_true")
    parser.add_argument(
        "--nearest_neighbors_metric",
        type=str,
        default="cos_sim",
        choices=["cos_sim", "dot_prod"],
    )
    parser.add_argument("--num_neighbors_for_negatives", type=int, default=4096)
    parser.add_argument("--train_sample_ratio", type=float, default=0.1)
    parser.add_argument("--group_size", type=int, default=500)

    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--bs_trn", default=128, type=int)
    parser.add_argument("--bs_val", default=128, type=int)
    parser.add_argument(
        "--optimizer", default="sgd", type=str, choices=["sgd", "adam", "adamw"]
    )
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)

    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--replicate", default=1, type=int)
    parser.add_argument("--no_cuda", default=False, action="store_true")
    parser.add_argument("--torch_seed", default=False, action="store_true")
    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--num_epochs_to_end", default=0, type=int)

    parser.add_argument("--avg_thshd", default=90.0, type=float)
    parser.add_argument("--wg_thshd", default=90.0, type=float)

    parser.add_argument(
        "--retest_dataset", type=str, default="imagenet", choices=RETEST_DATASETS
    )
    parser.add_argument("--retest_batch_size", default=128, type=int)

    args = parser.parse_args()

    args.arch = args.load_base_model.split("_")[-1].replace("/", "_").replace("-", "_")
    args.directory_name = "contrastive_adapter"
    args.zeroshot_predict_by = "kmeans_umap"

    if torch.cuda.is_available() and args.no_cuda is False:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    return args


def prepare_experiment(args):
    base_model_args = args.load_base_model.split("_")
    (
        base_model,
        base_transform,
        get_embeddings,
        get_dataset_embeddings,
        get_zeroshot_predictions,
    ) = load_base_model(base_model_args, args, clip=clip)

    load_dataloaders = initialize_data(args)
    dataloaders_base = load_dataloaders(
        args, train_shuffle=False, transform=base_transform
    )
    splits = ["train", "val", "test"]

    initialize_experiment(args)

    args.num_classes = len(args.train_classes)
    args.text_descriptions = update_classification_prompts(args)
    query_embeddings = get_embeddings(
        args.text_descriptions, base_model, args, normalize=True, verbose=True
    )

    dataset_embeddings = {}
    for dix, split in enumerate(splits):
        dataset_embeddings[split] = get_dataset_embeddings(
            base_model, dataloaders_base[dix], args, split=split
        )
    args.base_model_dim = dataset_embeddings["train"].shape[1]
    print(f"-> Embedding dimensions: {args.base_model_dim}")
    print("---")

    dataset_predictions = {}
    for dix, split in enumerate(splits):
        dataset_predictions[split] = get_zeroshot_predictions(
            dataset_embeddings[split],
            args.text_descriptions,
            args.zeroshot_predict_by,
            args,
            dataloaders_base[dix],
            temperature=100.0,
            split=split,
            numpy=True,
            base_model=base_model,
        )
    for ix, split in enumerate(splits):
        print("-" * len(split))
        print(f"Zero-shot {split} predictions")
        print("-" * len(split))
        evaluate_clip(dataset_predictions[split], dataloaders_base[ix], verbose=True)

    embeddings = [dataset_embeddings[split] for split in splits]
    indices = [[np.arange(len(e))] for e in embeddings]
    dataloaders = get_embedding_loaders_by_split(
        embeddings, indices, dataloaders_base, train_shuffle=False, args=args
    )
    dataloaders = [dataloader[0] for dataloader in dataloaders]

    base_model.cpu()

    return {
        "base_model": base_model,
        "get_zeroshot_predictions": get_zeroshot_predictions,
        "query_embeddings": query_embeddings,
        "dataloaders_base": dataloaders_base,
        "dataloaders": dataloaders,
        "splits": splits,
        "dataset_embeddings": dataset_embeddings,
        "dataset_predictions": dataset_predictions,
    }


def get_train_method(args):
    return args.train_method_verbose.replace("/", "_").replace("-", "_")


def get_adapter_embedding_path(args, suffix):
    train_method = get_train_method(args)
    prefix = "ERM_" if args.replicate == 4 else ""
    os.makedirs(paths.ADAPTER_CACHE_DIR, exist_ok=True)
    return os.path.join(
        paths.ADAPTER_CACHE_DIR,
        f"d={args.dataset}-s={args.seed}-sp=train-m={args.load_base_model}-{train_method}-o={args.optimizer}-me={args.max_epochs}-lr={args.lr}-bs_tr={args.bs_trn}-mo={args.momentum}-wd={args.weight_decay}-r={args.replicate}-s={args.seed}_{prefix}{suffix}.pt",
    )


def evaluate_saved_adapter(args, exp, model_path, suffix, title):
    model = get_adapter(args.num_classes, args, exp["query_embeddings"])
    print(f"=== {title} ===")
    model.load_state_dict(torch.load(model_path))

    test_metrics = None
    for dix, split in enumerate(exp["splits"]):
        print("-" * 5, split, "-" * 5)
        embedding_loader = exp["dataloaders"][dix]
        predictions = get_adapter_predictions(model, embedding_loader, args.device)
        avg_acc, wg_acc = evaluate_dataset_prediction(
            predictions, embedding_loader, args, verbose=True
        )
        if split == "test":
            test_metrics = (avg_acc, wg_acc)

    test_ix = exp["splits"].index("test")
    adapter_embeddings = get_adapter_embeddings(
        model, exp["dataloaders"][test_ix], args.device
    )
    embedding_path = get_adapter_embedding_path(args, suffix)
    torch.save(adapter_embeddings, embedding_path)
    print(f"-> Saved image embeddings to {embedding_path}!")

    predictions = exp["get_zeroshot_predictions"](
        adapter_embeddings,
        args.text_descriptions,
        "text",
        args,
        exp["dataloaders_base"][test_ix],
        temperature=100.0,
        split="test",
        numpy=True,
        base_model=exp["base_model"],
    )
    print("----")
    print("Zero-shot test predictions")
    print("----")
    evaluate_clip(predictions, exp["dataloaders_base"][test_ix], verbose=True)

    return model, test_metrics
