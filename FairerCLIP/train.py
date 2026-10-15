import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from base import AlternatingOptimizer
from clip import clip
from datasets import Clipfeature
from defaults import _C as cfg
from network import load_base_model
from network.clip import evaluate_clip

TEXT_DESCRIPTIONS = {
    "waterbirds": [
        "This is a picture of a landbird.",
        "This is a picture of a waterbird.",
    ],
    "celebA": [
        "A photo of a celebrity with dark hair.",
        "A photo of a celebrity with blond hair.",
    ],
}


def add_common_args(parser):
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
    parser.add_argument("--seed", default=2, type=int)


def setup_cfg(args):
    cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(
        "tau_i: %.2f tau_z_i: %.2f tau_t: %.2f tau_z_t: %.2f"
        % (cfg.tau_i, cfg.tau_z_i, cfg.tau_t, cfg.tau_z_t)
    )


def train_and_evaluate():
    trainset = Clipfeature("train", cfg)
    traindata = DataLoader(trainset, batch_size=trainset.__len__(), shuffle=False)

    testset = Clipfeature("test", cfg)
    testdata = DataLoader(testset, batch_size=testset.__len__(), shuffle=False)

    base_model_args = cfg.load_base_model.split("_")
    base_model, _, get_embeddings, get_zeroshot_predictions = load_base_model(
        base_model_args, cfg, clip=clip
    )

    text_descriptions = TEXT_DESCRIPTIONS[cfg.dataset]
    query_embeddings = get_embeddings(
        text_descriptions, base_model, cfg, normalize=True, verbose=False
    )
    text_embeddings = query_embeddings.float().to(cfg.device)

    imfeat_train, labels_train_y, labels_train_s, _ = next(iter(traindata))

    iters = cfg.iters if cfg.nolabel == True else 1
    model = AlternatingOptimizer(cfg)
    model.main(
        imfeat_train,
        labels_train_y,
        labels_train_s,
        labels_train_y,
        labels_train_s,
        text_embeddings,
        iters,
        get_zeroshot_predictions,
        cfg,
    )

    imfeat_test, _, labels_test_s, labels_test_y_gt = next(iter(testdata))
    debias_image = model.get_feat(imfeat_test)
    text_embeddings_debias = model.get_textfeat(text_embeddings)
    dataset_predictions = get_zeroshot_predictions(
        debias_image, text_embeddings_debias, cfg, temperature=100.0
    )

    print("result for testing set:")
    avg_acc, robust_acc, groups_acc = evaluate_clip(
        dataset_predictions, labels_test_y_gt, labels_test_s, verbose=True
    )

    return {
        "model": model,
        "base_model": base_model,
        "get_embeddings": get_embeddings,
        "get_zeroshot_predictions": get_zeroshot_predictions,
        "text_embeddings": text_embeddings,
        "avg_acc": avg_acc,
        "robust_acc": robust_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="fairerclip")
    add_common_args(parser)
    args = parser.parse_args()
    setup_cfg(args)
    train_and_evaluate()


if __name__ == "__main__":
    main()
