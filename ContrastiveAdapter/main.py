import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    evaluate_saved_adapter,
    get_args,
    get_train_method,
    prepare_experiment,
)
from datasets import balance_dataset_class_group, get_resampled_loader_by_pred_by_class
from datasets.contrastive_embeddings import load_supervised_contrastive_dataset
from datasets.embeddings import get_similarity_matrix
from network import get_adapter, get_optimizer
from train import evaluate, train_epoch
from train.contrastive import ContrastiveLoss, train_contrastive_epoch
from utils.logging import log_metrics, process_validation_metrics


def main():
    args = get_args()
    np.random.seed(args.seed)
    if args.torch_seed:
        torch.manual_seed(args.seed)

    exp = prepare_experiment(args)
    if args.zero_shot is True:
        return

    dataset_embeddings = exp["dataset_embeddings"]
    dataset_predictions = exp["dataset_predictions"]
    train_loader, val_loader, test_loader = exp["dataloaders"]
    num_classes = args.num_classes

    classifier_criterion = nn.CrossEntropyLoss()
    contrastive_criterion = ContrastiveLoss(
        args.temperature, args.num_negatives, args.num_positives
    )
    contrastive_args = {
        "source_predictions": dataset_predictions["train"],
        "source_embeddings": dataset_embeddings["train"],
        "source_dataloader": train_loader,
        "negative_indices_by_class": None,
        "adapter": None,
    }
    contrastive_args["source_embeddings_normed"] = contrastive_args[
        "source_embeddings"
    ] / contrastive_args["source_embeddings"].norm(dim=-1, keepdim=True)

    try:
        similarity_matrix = get_similarity_matrix(
            contrastive_args["source_embeddings_normed"],
            contrastive_args["source_embeddings_normed"],
            contrastive_args["source_dataloader"],
            metric="cos_sim",
            args=args,
            batch_size=16384 * 2,
            split="train",
        )
    except:
        similarity_matrix = None
    contrastive_args["embedding_similarity_matrix"] = similarity_matrix

    args.best_loss = 1e10
    args.best_loss_epoch = -1
    args.best_avg_acc = -1
    args.best_avg_acc_epoch = -1
    args.best_robust_acc = -1
    args.best_robust_acc_epoch = -1

    model = get_adapter(num_classes, args, exp["query_embeddings"])
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    args.num_adapter_parameters = sum([np.prod(p.size()) for p in model_parameters])
    optimizer = get_optimizer(model, args)

    print(f"-> Base zero-shot model = {args.load_base_model}")
    print(f"   - number of parameters:   {args.num_base_model_parameters}")
    print(f"-> Adapter train method = {args.train_method}")
    print(f"   - number of parameters:   {args.num_adapter_parameters}")
    print(f"   - input_dim:              {args.base_model_dim}")
    print(f"   - hidden_dim:             {args.hidden_dim}")
    print(f"   - batch_norm:             {args.adapter_head_batch_norm}")
    print(f"   - residual connection:    {args.residual_connection}")
    print(f"   - activation function:    {args.activation}")
    print(f"-> Classification prompts ({args.num_classes}):")
    for prompt in args.text_descriptions:
        print(f"   - {prompt}")
    print(f"-> Experiment: {args.experiment_name}")

    train_loader_ = DataLoader(
        train_loader.dataset,
        shuffle=True,
        batch_size=args.bs_trn,
        num_workers=args.num_workers,
    )

    try:
        dataset_predictions["train"] = dataset_predictions["train"].numpy()
    except:
        pass

    train_method = get_train_method(args)
    pbar = tqdm(range(args.max_epochs))

    best_metric_counter = 0
    stopped_early = False
    train_m = None
    val_m = None

    for epoch in pbar:
        if epoch == 0:
            pbar.set_description(f"Epoch: {epoch}")
        else:
            train_loss, train_avg_acc, train_min_acc = train_m
            val_loss, val_avg_acc, val_min_acc = val_m
            pbar.set_description(
                f"Epoch: {epoch} | Train Loss: {train_loss:.3f}, Avg Acc: {train_avg_acc:.2f}%, Robust Acc: {train_min_acc:.2f}% | Val Loss: {val_loss:.3f}, Avg Acc: {val_avg_acc:.2f}%, Robust Acc: {val_min_acc:.2f}%, Best Avg Acc: {args.best_avg_acc:.2f}%, Best Robust Acc: {args.best_robust_acc:.2f}%, Best Robust Epoch: {args.best_robust_acc_epoch}"
            )

        if args.replicate == 1 and (epoch + 1) % 2 == 0:
            train_loader_rs = get_resampled_loader_by_pred_by_class(
                dataset_predictions["train"],
                train_loader.dataset,
                resample="group_size",
                seed=epoch + args.seed,
                args=args,
                batch_size=128,
            )
            train_epoch(
                model,
                train_loader_rs,
                optimizer,
                classifier_criterion,
                args,
                evaluate=False,
            )
            del train_loader_rs

        elif args.replicate == 2 and (epoch + 1) % 2 == 0:
            dataset_rs = balance_dataset_class_group(
                dataset_predictions["train"], train_loader.dataset, epoch + args.seed
            )
            drop_last = len(dataset_rs) % args.bs_trn == 1
            train_loader_rs = DataLoader(
                dataset_rs,
                batch_size=args.bs_trn,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=drop_last,
            )
            train_epoch(
                model,
                train_loader_rs,
                optimizer,
                classifier_criterion,
                args,
                evaluate=False,
            )
            del train_loader_rs

        elif args.replicate == 4:
            train_epoch(model, train_loader_, optimizer, classifier_criterion, args)

        else:
            contrastive_train_loader = load_supervised_contrastive_dataset(
                args, epoch + args.seed, **contrastive_args
            )
            train_contrastive_epoch(
                model,
                optimizer,
                classifier_criterion,
                contrastive_criterion,
                contrastive_train_loader,
                args,
            )
            del contrastive_train_loader

        train_metrics = evaluate(model, train_loader, classifier_criterion, args)
        val_metrics = evaluate(model, val_loader, classifier_criterion, args)
        test_metrics = evaluate(model, test_loader, classifier_criterion, args)

        train_m, val_m = log_metrics(
            train_metrics, val_metrics, test_metrics, epoch, dataset_ix=0, args=args
        )
        loss_m, avg_acc_m, min_acc_m = process_validation_metrics(
            model,
            val_metrics,
            epoch,
            train_method,
            args,
            best_loss=args.best_loss,
            best_avg_acc=args.best_avg_acc,
            best_robust_acc=args.best_robust_acc,
            best_loss_epoch=args.best_loss_epoch,
            best_avg_acc_epoch=args.best_avg_acc_epoch,
            best_robust_acc_epoch=args.best_robust_acc_epoch,
        )

        if min_acc_m[-1] != args.best_robust_acc_epoch:
            best_metric_counter = 0
        else:
            best_metric_counter += 1

        args.best_loss, args.best_loss_epoch = loss_m
        args.best_avg_acc, args.best_avg_acc_epoch = avg_acc_m
        args.best_robust_acc, args.best_robust_acc_epoch = min_acc_m

        if (epoch + 1) % 10 == 0:
            df = pd.DataFrame(args.results_dict)
            df.to_csv(args.results_path)

        if best_metric_counter > args.num_epochs_to_end and args.num_epochs_to_end != 0:
            stopped_early = True
            break

        if val_m[1] > 95 and val_m[2] > 85:
            stopped_early = True
            break

        val_avg_now = val_m[1]
        val_wg_now = val_m[2]
        if val_avg_now > args.avg_thshd and val_wg_now > args.wg_thshd:
            stopped_early = True
            print(
                f"Now, Avg={round(val_avg_now,3)} and WG={round(val_wg_now,3)} so we stop training."
            )
            break

    if stopped_early:
        print(f"-> Stopping experiment after {epoch + 1} epochs...")

    df = pd.DataFrame(args.results_dict)
    df.to_csv(args.results_path)

    max_robust_acc_ix = np.argmax(df["val_robust_acc"])
    max_epoch = df["epoch"].iloc[max_robust_acc_ix]
    print(f"-> Best validation metrics (Epoch {max_epoch}):")
    print(
        df[["val_avg_acc", "val_robust_acc", "test_avg_acc", "test_robust_acc"]].iloc[
            max_robust_acc_ix
        ]
    )

    evaluate_saved_adapter(
        args,
        exp,
        args.best_robust_acc_model_path,
        "WG",
        "Best model by robust validation performance",
    )
    evaluate_saved_adapter(
        args,
        exp,
        args.best_avg_acc_model_path,
        "AVG",
        "Best model by average validation performance",
    )


if __name__ == "__main__":
    main()
