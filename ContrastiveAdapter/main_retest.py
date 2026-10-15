import csv
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from clip import clip
from common import (
    evaluate_saved_adapter,
    get_args,
    get_train_method,
    prepare_experiment,
)
from configs import paths
from datasets.datasets import _transform, get_retest_prompts, load_retest_dataset
from evaluate import get_adapter_embeddings_v2
from network.clip import classify_embeddings_with_text_v2
from utils.logging import get_model_path


def evaluate_nontarget(args, clip_model, encoding_coll, label_coll, retest_prompt):
    all_encodings = torch.cat(encoding_coll, dim=0).to(args.device)
    all_labels = torch.cat(label_coll, dim=0).to(args.device)

    with torch.no_grad():
        retest_predictions = classify_embeddings_with_text_v2(
            clip_model, all_encodings, retest_prompt, args, temperature=100.0
        )

    if isinstance(retest_predictions, np.ndarray):
        retest_predictions = torch.from_numpy(retest_predictions).to(args.device)
    retest_predictions = retest_predictions.float()

    correct = (retest_predictions == all_labels).sum().item()
    total = all_labels.size(0)
    return correct / total * 100


def main():
    args = get_args()
    np.random.seed(args.seed)
    if args.torch_seed:
        torch.manual_seed(args.seed)

    print("------------------------------------")
    print(f"Model: {args.load_base_model}")
    print(f"Target Dataset: {args.dataset}")
    print(f"Non-target Dataset: {args.retest_dataset}")
    print("------------------------------------")

    exp = prepare_experiment(args)
    zeroshot_clip_model = exp["base_model"]

    train_method = get_train_method(args)
    args.best_robust_acc_model_path = get_model_path(args, "m-best_acc_r", train_method)
    args.best_avg_acc_model_path = get_model_path(args, "m-best_acc", train_method)

    model, (adj_avg_acc, robust_acc) = evaluate_saved_adapter(
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

    model_tag = args.load_base_model.lower().replace("/", "")
    os.makedirs(paths.NON_TARGET_CLASS_CACHE_DIR, exist_ok=True)
    precomputed_encs_file = os.path.join(
        paths.NON_TARGET_CLASS_CACHE_DIR, f"{args.retest_dataset}_{model_tag}.pkl"
    )

    adapter_cache_name = (
        "precomputed_encs_adapter_erm"
        if args.replicate == 4
        else "precomputed_encs_adapter"
    )
    precomputed_encs_folder_ada = os.path.join(
        paths.ADAPTER_CACHE_DIR, adapter_cache_name
    )
    os.makedirs(precomputed_encs_folder_ada, exist_ok=True)
    precomputed_encs_file_ada = os.path.join(
        precomputed_encs_folder_ada,
        f"{args.dataset}_{args.retest_dataset}_{model_tag}_seed={args.seed}-{train_method}-o={args.optimizer}-me={args.max_epochs}-lr={args.lr}-bs_tr={args.bs_trn}-mo={args.momentum}-wd={args.weight_decay}-r={args.replicate}-s={args.seed}_adapter.pkl",
    )

    retest_prompt = get_retest_prompts(args.retest_dataset)

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

        encoder, preprocess = clip.load(
            args.load_base_model.split("_")[1], device=args.device, jit=False
        )
        encoder.eval()
        encoder.requires_grad_(False)

        with torch.no_grad():
            for images, labels in tqdm(
                dataloader, desc="Precomputing image embeddings..."
            ):
                images = images.to(args.device)
                label_coll.append(labels)
                image_encodings = F.normalize(encoder.encode_image(images))
                enc_coll.append(image_encodings.cpu())
        load_res = {"enc": enc_coll, "labels": label_coll}
        pickle.dump(load_res, open(precomputed_encs_file, "wb"))

    encoding_coll = load_res["enc"]
    label_coll = load_res["labels"]

    if os.path.exists(precomputed_encs_file_ada):
        load_res_ada = pickle.load(open(precomputed_encs_file_ada, "rb"))
    else:
        adapter_embeddings = get_adapter_embeddings_v2(
            model, encoding_coll, label_coll, args.device
        )
        load_res_ada = {"enc": adapter_embeddings, "labels": label_coll}
        pickle.dump(load_res_ada, open(precomputed_encs_file_ada, "wb"))

    encoding_coll_ada = load_res_ada["enc"]
    label_coll_ada = load_res_ada["labels"]

    acc_zeroshot = evaluate_nontarget(
        args, zeroshot_clip_model, encoding_coll, label_coll, retest_prompt
    )
    print(f"Non-target Accuracy (Zero-shot CLIP): {acc_zeroshot:.2f} %")

    acc_adapter = evaluate_nontarget(
        args, zeroshot_clip_model, encoding_coll_ada, label_coll_ada, retest_prompt
    )
    print(f"Non-target Accuracy (Adapter): {acc_adapter:.2f} %")

    result_dir_path = f"./Result/{args.dataset}/"
    os.makedirs(result_dir_path, exist_ok=True)
    csv_file_path = f"{result_dir_path}/result_r={args.replicate}.csv"
    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "target classes",
                    "seed",
                    "replicate",
                    "WG",
                    "Avg",
                    "Gap",
                    "non-target classes",
                    "Acc (zero-shot)",
                    "Acc (adapter)",
                ]
            )
        writer.writerow(
            [
                args.dataset,
                args.seed,
                args.replicate,
                round(robust_acc, 1),
                round(adj_avg_acc, 1),
                round(adj_avg_acc - robust_acc, 1),
                args.retest_dataset,
                round(acc_zeroshot, 1),
                round(acc_adapter, 1),
            ]
        )


if __name__ == "__main__":
    main()
