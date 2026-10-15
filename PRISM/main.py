import argparse
import csv
import os
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from wilds.common.data_loaders import get_train_loader

import clip
from configs import paths
from data.CelebA import CelebADataset
from data.SSL_text import get_SSL_dataset
from data.Waterbird import WaterbirdsDataset
from data.datasets import (
    RETEST_DATASETS,
    _transform,
    get_retest_prompts,
    load_retest_dataset,
)
from models import get_transformer
from utils import (
    accuracy_by_subgroup,
    classify_images,
    encode_text,
    evaluate_nontarget,
    print_per_class,
    train_transformation,
)


def get_args():
    parser = argparse.ArgumentParser(description="PRISM")

    parser.add_argument("--dataset", type=str.lower, default="waterbirds")
    parser.add_argument("--CLIP_model", type=str, default="ViT-L/14")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--per_group", default=False, action="store_true")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--init_weight", type=str.lower, default="random")
    parser.add_argument("--reg_type", type=str.lower, default=None)
    parser.add_argument("--reg_lambda", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cache_P", default=False, action="store_true")

    parser.add_argument(
        "--retest_dataset", type=str, default="imagenet", choices=RETEST_DATASETS
    )
    parser.add_argument("--retest_batch_size", type=int, default=128)

    args = parser.parse_args()
    if args.reg_type in (None, "none"):
        args.reg_type = None
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.model_name = args.CLIP_model.replace("/", "")
    return args


def print_info(args):
    print("------------------------------------")
    print(f"Model: {args.CLIP_model}")
    print(f"Target Dataset: {args.dataset}")
    print(f"Non-target Dataset: {args.retest_dataset}")
    print("------------------------------------")


def main():
    args = get_args()
    print_info(args)
    torch.manual_seed(args.seed)

    model, preprocess = clip.load(args.CLIP_model, device=args.device)

    if args.dataset == "celeba":
        dataset = CelebADataset(root_dir=paths.CELEBA_DIR, download=True)
        text_descriptions = [
            "A photo of a celebrity with dark hair.",
            "A photo of a celebrity with blond hair.",
        ]
    elif args.dataset == "waterbirds":
        dataset = WaterbirdsDataset(root_dir=paths.WATERBIRDS_DIR, download=True)
        text_descriptions = [
            "This is a picture of a landbird.",
            "This is a picture of a waterbird.",
        ]

    test_data = dataset.get_subset("test", transform=preprocess)
    test_loader = get_train_loader(
        "standard", test_data, batch_size=args.batch_size, num_workers=args.num_workers
    )

    query_embeddings = encode_text(model, text_descriptions, args.device)

    text_loader = get_SSL_dataset(args)
    transformer = get_transformer(args, embed_dim=query_embeddings.shape[-1])

    os.makedirs(paths.MODEL_CACHE_DIR, exist_ok=True)
    transformer_path = os.path.join(
        paths.MODEL_CACHE_DIR,
        f"{args.dataset}_{args.model_name}_ep={args.epochs}_ns={args.num_samples}_lr={args.lr}_wd={args.wd}_iw={args.init_weight}_reg_type={args.reg_type}_reg_lambda={args.reg_lambda}_seed={args.seed}.pth",
    )
    if args.use_cache_P and os.path.exists(transformer_path):
        print(f"Loading trained transformer from {transformer_path}")
        transformer.load_state_dict(
            torch.load(transformer_path, map_location=args.device)
        )
    else:
        print("Training transformer...")
        train_transformation(args, model, text_loader, transformer)
        torch.save(transformer.state_dict(), transformer_path)
    P = transformer.transformer.weight.data.clone().T

    P = P.to(query_embeddings.dtype)

    _, [all_y, all_preds, all_metadata] = classify_images(
        args, model, query_embeddings, test_loader, P=P, description="Classifying"
    )

    eval_results = dataset.eval(
        all_preds.to("cpu"), all_y.to("cpu"), all_metadata.to("cpu")
    )
    results_dict = eval_results[0]

    if args.dataset == "celeba":
        adj_acc_avg = results_dict["acc_avg"]
    elif args.dataset == "waterbirds":
        adj_acc_avg = results_dict["adj_acc_avg"]
    worst_group_acc = results_dict["acc_wg"]

    print(f"Adjusted Average Accuracy: {adj_acc_avg:.3f}")
    print(f"Worst-Group Accuracy: {worst_group_acc:.3f}")

    robust_acc = worst_group_acc * 100
    adj_avg_acc = adj_acc_avg * 100
    gap = adj_avg_acc - robust_acc

    if args.per_group:
        print_per_class(args, eval_results)
        accuracy_by_subgroup(
            list(all_preds.to("cpu").numpy()),
            list(all_y.to("cpu").numpy()),
            [x[0] for x in list(all_metadata.to("cpu").numpy())],
        )

    retest_prompt = get_retest_prompts(args.retest_dataset)

    os.makedirs(paths.NON_TARGET_CLASS_CACHE_DIR, exist_ok=True)
    precomputed_encs_file = os.path.join(
        paths.NON_TARGET_CLASS_CACHE_DIR,
        f"{args.retest_dataset}_{args.model_name}.pkl",
    )

    if os.path.exists(precomputed_encs_file):
        load_res = pickle.load(open(precomputed_encs_file, "rb"))
    else:
        retest_dataset = load_retest_dataset(args.retest_dataset, _transform(224))
        dataloader = DataLoader(
            retest_dataset,
            args.retest_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        enc_coll = []
        label_coll = []
        with torch.no_grad():
            for images, labels in tqdm(
                dataloader, desc="Precomputing image embeddings..."
            ):
                images = images.to(args.device)
                label_coll.append(labels)
                image_encodings = F.normalize(model.encode_image(images.half()))
                enc_coll.append(image_encodings.cpu())
        load_res = {"enc": enc_coll, "labels": label_coll}
        pickle.dump(load_res, open(precomputed_encs_file, "wb"))

    encoding_coll = load_res["enc"]
    label_coll = load_res["labels"]

    text_embeddings = encode_text(model, retest_prompt, args.device).float()

    acc_zeroshot = evaluate_nontarget(args, encoding_coll, label_coll, text_embeddings)
    print(f"Non-target Accuracy (Zero-shot CLIP): {acc_zeroshot:.2f} %")

    acc_cali = evaluate_nontarget(args, encoding_coll, label_coll, text_embeddings, P=P)
    print(f"Non-target Accuracy (PRISM): {acc_cali:.2f} %")

    result_dir_path = f"./Result/{args.dataset}/{args.model_name}"
    os.makedirs(result_dir_path, exist_ok=True)
    csv_file_path = f"{result_dir_path}/result.csv"
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "target classes",
                    "seed",
                    "WG",
                    "Avg",
                    "Gap",
                    "non-target classes",
                    "Acc (zero-shot)",
                    "Acc (PRISM)",
                ]
            )
        writer.writerow(
            [
                args.dataset,
                args.seed,
                round(robust_acc, 1),
                round(adj_avg_acc, 1),
                round(gap, 1),
                args.retest_dataset,
                round(acc_zeroshot, 1),
                round(acc_cali, 1),
            ]
        )


if __name__ == "__main__":
    main()
