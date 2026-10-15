import os
import numpy as np
import torch

import umap

from tqdm import tqdm
from sklearn.cluster import KMeans

from clip import clip

from utils.logging import summarize_acc


def get_embeddings(text, clip_model, args, normalize=True, verbose=True):
    if verbose:
        desc = "-> Text descriptions for zero-shot classification:"
        print("-" * len(desc))
        print(desc)
        num_display = 5
        for d in text[:num_display]:
            print(f"   - {d}")
        if len(text) > num_display:
            print("     ...")
            for d in text[-num_display:]:
                print(f"   - {d}")
        print("-" * len(desc))
    text_tokens = clip.tokenize(text)
    clip_model.to(args.device)
    clip_model.eval()
    with torch.no_grad():
        text_tokens = text_tokens.to(args.device)
        text_embeddings = clip_model.encode_text(text_tokens).float().cpu()
        if normalize:
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    clip_model.cpu()
    return text_embeddings


def get_dataset_embeddings(model, dataloader, args, split="train"):
    return get_clip_embeddings(model, dataloader, args, split)


def get_clip_embeddings(model, dataloader, args, split="train", verbose=False):
    verbose = True if args.verbose else False

    embedding_fname = f"d={args.dataset}-s={split}-m={args.load_base_model}.pt"
    embedding_path = os.path.join(args.embeddings_dir, embedding_fname)

    if os.path.exists(embedding_path):
        if verbose:
            print(f"-> Retrieving image embeddings from {embedding_path}!")
        embeddings = torch.load(embedding_path)
        return embeddings
    else:
        if verbose:
            print(f"-> Image embeddings from {embedding_path} not found.")

    model.to(args.device)
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for ix, data in enumerate(
            tqdm(
                dataloader,
                desc=f"Computing {args.load_base_model} image embeddings for {split} split",
            )
        ):
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            embeddings = model.encode_image(inputs).float().cpu()
            all_embeddings.append(embeddings)
            inputs = inputs.cpu()
    model.cpu()

    torch.save(torch.cat(all_embeddings), embedding_path)
    if verbose:
        print(f"-> Saved image embeddings to {embedding_path}!")

    return torch.cat(all_embeddings)


def get_umap_embeddings(embeddings, seed, split="train", args=None):
    embedding_fname = f"umap-d={args.dataset}-split={split}-c={args.config}-m={args.load_base_model}-s={seed}.npy"
    embedding_path = os.path.join(args.embeddings_dir, embedding_fname)
    if os.path.exists(embedding_path):
        print(f"-> Retrieving UMAP embeddings from {embedding_path}!")
        return np.load(embedding_path)
    else:
        print(f"-> UMAP embeddings from {embedding_path} not found.")
        print(f"-> Computing UMAP of embedding size {[s for s in embeddings.shape]}...")

        embeddings = umap.UMAP(random_state=seed).fit_transform(embeddings)
        np.save(embedding_path, embeddings)
        return embeddings


def evaluate_clip(clip_predictions, dataloader, verbose=False):
    targets_t = dataloader.dataset.targets_all["target"].astype(int)
    targets_s = dataloader.dataset.targets_all["spurious"].astype(int)

    correct_by_groups = np.zeros([len(np.unique(targets_t)), len(np.unique(targets_s))])
    total_by_groups = np.zeros(correct_by_groups.shape)

    correct = clip_predictions == targets_t

    for ix, y in enumerate(targets_t):
        s = targets_s[ix]
        correct_by_groups[int(y)][int(s)] += correct[ix].item()
        total_by_groups[int(y)][int(s)] += 1

    avg_acc, robust_acc = summarize_acc(
        correct_by_groups, total_by_groups, stdout=verbose
    )

    return avg_acc, robust_acc


def classify_embeddings_with_text(
    clip_model, image_embeddings, text_descriptions, args, temperature=100.0
):
    text_tokens = clip.tokenize(text_descriptions)
    clip_model.to(args.device)
    clip_model.eval()
    with torch.no_grad():
        _image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )

        text_tokens = text_tokens.to(args.device)
        text_embeddings = clip_model.encode_text(text_tokens).float().cpu()
        _text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        cross = _image_embeddings @ _text_embeddings.T
        text_probs = (temperature * cross).softmax(dim=-1)
        _, predicted = torch.max(text_probs.data, 1)

        text_tokens = text_tokens.cpu()

    clip_model.cpu()
    return predicted.cpu().numpy()


def classify_embeddings_with_text_v2(
    clip_model, image_embeddings, text_descriptions, args, temperature=100.0
):
    text_tokens = clip.tokenize(text_descriptions).to(args.device)
    clip_model.to(args.device)
    clip_model.eval()

    with torch.no_grad():
        _image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )

        text_embeddings = clip_model.encode_text(text_tokens).float()
        _text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        _image_embeddings = _image_embeddings.to(args.device)
        _text_embeddings = _text_embeddings.to(args.device)

        cross = _image_embeddings.float() @ _text_embeddings.T
        text_probs = (temperature * cross).softmax(dim=-1)
        _, predicted = torch.max(text_probs.data, 1)

    clip_model.cpu()
    return predicted.cpu().numpy()


def classify_embeddings_with_clustering(embeddings, labels, num_classes, seed):
    kmeans = KMeans(n_clusters=num_classes, random_state=seed)
    kmeans.fit(embeddings)
    predicted = kmeans.labels_

    correct_0 = (predicted == labels).sum()
    correct_1 = (1 - predicted == labels).sum()
    if correct_1 > correct_0:
        predicted = 1 - predicted
    return predicted


def get_zeroshot_predictions(
    key_embeddings,
    text_descriptions,
    predict_by,
    args,
    dataloader,
    temperature=100.0,
    split="train",
    numpy=True,
    base_model=None,
):
    if predict_by == "text":
        predictions = classify_embeddings_with_text(
            base_model, key_embeddings, text_descriptions, args, temperature=100.0
        )
    else:
        if "umap" in predict_by:
            print("Computing UMAP...")
            key_embeddings = get_umap_embeddings(
                key_embeddings, seed=args.seed, split=split, args=args
            )
        predictions = classify_embeddings_with_clustering(
            key_embeddings,
            dataloader.dataset.targets_all["target"],
            args.num_classes,
            args.seed,
        )

    return predictions
