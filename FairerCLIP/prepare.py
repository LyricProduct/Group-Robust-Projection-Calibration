import argparse
import csv
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from clip import clip
from configs import paths
from defaults import _C as cfg
from network import load_base_model

SPLIT_IDS = {"train": 0, "test": 2}

CSV_TAGS = {"waterbirds": "waterbirds", "celebA": "celeba"}

S_TEXT_DESCRIPTIONS = {
    "waterbirds": [
        "This is a picture of a land background.",
        "This is a picture of a water background.",
    ],
    "celebA": [
        "This is a photo of a male celebrity.",
        "This is a photo of a female celebrity.",
    ],
}

Y_TEXT_DESCRIPTIONS = {
    "waterbirds": [
        "This is a picture of a landbird.",
        "This is a picture of a waterbird.",
    ],
    "celebA": [
        "A photo of a celebrity with dark hair.",
        "A photo of a celebrity with blond hair.",
    ],
}


def load_split(dataset, split_type):
    if dataset == "waterbirds":
        data_dir = os.path.join(
            paths.WATERBIRDS_DIR, "waterbird_complete95_forest2water2"
        )
        df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
        df = df[df["split"] == SPLIT_IDS[split_type]]
        image_paths = [os.path.join(data_dir, x) for x in df["img_filename"]]
    elif dataset == "celebA":
        attributes = pd.read_csv(os.path.join(paths.CELEBA_DIR, "list_attr_celeba.csv"))
        partition = pd.read_csv(
            os.path.join(paths.CELEBA_DIR, "list_eval_partition.csv")
        )
        df = attributes[partition["partition"] == SPLIT_IDS[split_type]]
        image_paths = [
            os.path.join(paths.CELEBA_DIR, "img_align_celeba", x)
            for x in df["image_id"]
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return df, image_paths


def encode_prompts(model, prompts, device):
    text_input = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def extract_features(model, preprocess, device, image_paths, batch_size):
    all_image_features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_images = []
        for path in image_paths[i : i + batch_size]:
            image = Image.open(path).convert("RGB")
            batch_images.append(preprocess(image))

        image_input = torch.stack(batch_images).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        all_image_features.append(image_features.cpu())

    return torch.cat(all_image_features)


def write_predictions(path, predictions):
    with open(path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([[int(p)] for p in predictions])


def generate(
    model,
    preprocess,
    base_model_name,
    device,
    dataset,
    s_prompts,
    y_prompts,
    save_dir,
    split_type,
    batch_size=128,
):
    df, image_paths = load_split(dataset, split_type)

    s_text_features = encode_prompts(model, s_prompts, device)
    y_text_features = encode_prompts(model, y_prompts, device)

    imfeat = extract_features(model, preprocess, device, image_paths, batch_size)

    imfeat_path = os.path.join(
        save_dir, f"d={dataset}-s={split_type}-m={base_model_name}.pt"
    )
    torch.save(imfeat, imfeat_path)

    df.to_csv(
        os.path.join(save_dir, f"{CSV_TAGS[dataset]}_{split_type}.csv"),
        index=False,
        header=False,
    )

    if split_type == "train":
        with torch.no_grad():
            image_features = imfeat.to(device).to(s_text_features.dtype)
            s_predictions = (image_features @ s_text_features.t()).argmax(dim=1)
            y_predictions = (image_features @ y_text_features.t()).argmax(dim=1)
        write_predictions(
            os.path.join(
                save_dir, f"prediction-{CSV_TAGS[dataset]}-s-{split_type}.csv"
            ),
            s_predictions.cpu().numpy(),
        )
        write_predictions(
            os.path.join(
                save_dir, f"prediction-{CSV_TAGS[dataset]}-y-{split_type}.csv"
            ),
            y_predictions.cpu().numpy(),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fairerclip")
    parser.add_argument(
        "--config-file",
        default="configs/debias_waterbird.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs="+",
    )
    parser.add_argument("--save_dir", type=str, default=paths.FEATURES_DIR)

    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    base_model_args = cfg.load_base_model.split("_")
    base_model, base_transform, _, _ = load_base_model(base_model_args, cfg, clip=clip)
    base_model = base_model.to(cfg.device)

    for split_type in ["train", "test"]:
        generate(
            base_model,
            base_transform,
            cfg.load_base_model,
            cfg.device,
            cfg.dataset,
            S_TEXT_DESCRIPTIONS[cfg.dataset],
            Y_TEXT_DESCRIPTIONS[cfg.dataset],
            save_dir=args.save_dir,
            split_type=split_type,
        )
