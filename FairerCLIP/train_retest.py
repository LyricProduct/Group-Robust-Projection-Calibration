import argparse
import csv
import io
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import paths
from datasets.datasets import (
    RETEST_DATASETS,
    _transform,
    get_retest_prompts,
    load_retest_dataset,
)
from defaults import _C as cfg
from train import add_common_args, setup_cfg, train_and_evaluate


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(
                io.BytesIO(data), map_location="cpu", weights_only=False
            )
        return super().find_class(module, name)


def load_tensor_cache(path):
    with open(path, "rb") as handle:
        return CPUUnpickler(handle).load()


def evaluate_nontarget(
    encoding_coll, label_coll, text_embeddings, get_zeroshot_predictions, desc
):
    correct = 0
    total = 0
    for image_encodings, labels in tqdm(
        zip(encoding_coll, label_coll), total=len(encoding_coll), desc=desc
    ):
        image_encodings = image_encodings.to(cfg.device)
        labels = labels.to(cfg.device)

        retest_predictions = get_zeroshot_predictions(
            image_encodings.to(torch.float32), text_embeddings, cfg, temperature=100.0
        )
        if isinstance(retest_predictions, np.ndarray):
            retest_predictions = torch.from_numpy(retest_predictions).to(cfg.device)
        retest_predictions = retest_predictions.float()

        correct += (retest_predictions == labels).sum().item()
        total += labels.size(0)

    return correct / total * 100


def main():
    parser = argparse.ArgumentParser(description="fairerclip")
    add_common_args(parser)
    parser.add_argument(
        "--retest_dataset", type=str, default="imagenet", choices=RETEST_DATASETS
    )
    parser.add_argument("--retest_batch_size", default=128, type=int)
    args = parser.parse_args()
    setup_cfg(args)

    print("------------------------------------")
    print(f"Model: {cfg.load_base_model}")
    print(f"Target Dataset: {cfg.dataset}")
    print(f"Non-target Dataset: {args.retest_dataset}")
    print(f"Labels: {'without' if cfg.nolabel else 'with'}")
    print("------------------------------------")

    results = train_and_evaluate()
    model = results["model"]
    base_model = results["base_model"]
    get_embeddings = results["get_embeddings"]
    get_zeroshot_predictions = results["get_zeroshot_predictions"]
    text_embeddings = results["text_embeddings"]

    clip_tag = cfg.load_base_model.split("_")[1].lower()
    sa_labels = "wolabels" if cfg.nolabel else "wlabels"

    os.makedirs(paths.NON_TARGET_CLASS_CACHE_DIR, exist_ok=True)
    precomputed_encs_file = os.path.join(
        paths.NON_TARGET_CLASS_CACHE_DIR,
        f"{args.retest_dataset}_{clip_tag}.pkl",
    )

    os.makedirs(paths.NON_TARGET_KERNEL_CACHE_DIR, exist_ok=True)
    precomputed_encs_file_kernel = os.path.join(
        paths.NON_TARGET_KERNEL_CACHE_DIR,
        f"{cfg.dataset}_{args.retest_dataset}_{clip_tag}_{sa_labels}_seed{args.seed}_kernel.pkl",
    )

    retest_prompt = get_retest_prompts(args.retest_dataset)

    if os.path.exists(precomputed_encs_file):
        load_res = load_tensor_cache(precomputed_encs_file)
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

        base_model = base_model.to(cfg.device)
        base_model.eval()
        base_model.requires_grad_(False)

        with torch.no_grad():
            for images, labels in tqdm(
                dataloader, desc="Precomputing image embeddings..."
            ):
                images = images.to(cfg.device)
                label_coll.append(labels)
                image_encodings = F.normalize(base_model.encode_image(images.half()))
                enc_coll.append(image_encodings.cpu())
        load_res = {"enc": enc_coll, "labels": label_coll}
        with open(precomputed_encs_file, "wb") as handle:
            pickle.dump(load_res, handle)

    encoding_coll = load_res["enc"]
    label_coll = load_res["labels"]

    if os.path.exists(precomputed_encs_file_kernel):
        load_res_kernel = load_tensor_cache(precomputed_encs_file_kernel)
    else:
        kernel_img_embs = []
        for image_encodings in tqdm(
            encoding_coll, desc="Computing image embeddings with kernel ..."
        ):
            debias_image = model.image_model.encod(image_encodings.to(cfg.device))
            kernel_img_embs.append(debias_image.to(torch.float32).cpu())

        load_res_kernel = {"enc": kernel_img_embs, "labels": label_coll}
        with open(precomputed_encs_file_kernel, "wb") as handle:
            pickle.dump(load_res_kernel, handle)

    encoding_coll_kernel = load_res_kernel["enc"]
    label_coll_kernel = load_res_kernel["labels"]

    query_embeddings = get_embeddings(
        retest_prompt, base_model, cfg, normalize=True, verbose=False
    )
    text_embeddings = query_embeddings.float().to(cfg.device)

    acc_zeroshot = evaluate_nontarget(
        encoding_coll,
        label_coll,
        text_embeddings,
        get_zeroshot_predictions,
        "Retest with zero-shot CLIP",
    )
    print(f"Non-target Accuracy (Zero-shot CLIP): {acc_zeroshot:.2f} %")

    text_embeddings_debias = model.get_textfeat(text_embeddings)
    acc_kernel = evaluate_nontarget(
        encoding_coll_kernel,
        label_coll_kernel,
        text_embeddings_debias,
        get_zeroshot_predictions,
        "Retest with FairerCLIP",
    )
    print(f"Non-target Accuracy (FairerCLIP): {acc_kernel:.2f} %")

    result_dir_path = f"./Result/{cfg.dataset}/"
    os.makedirs(result_dir_path, exist_ok=True)
    csv_file_path = f"{result_dir_path}/result_{sa_labels}.csv"
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
                    "Acc (FairerCLIP)",
                ]
            )
        writer.writerow(
            [
                cfg.dataset,
                args.seed,
                round(results["robust_acc"], 1),
                round(results["avg_acc"], 1),
                round(results["avg_acc"] - results["robust_acc"], 1),
                args.retest_dataset,
                round(acc_zeroshot, 1),
                round(acc_kernel, 1),
            ]
        )


if __name__ == "__main__":
    main()
