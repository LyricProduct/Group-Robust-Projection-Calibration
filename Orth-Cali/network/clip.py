import os
import numpy as np
import torch
from tqdm import tqdm
from clip import clip
from utils.logging import summarize_acc
from PIL import ImageFile
from configs import paths

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_embeddings(text, clip_model, args, normalize=True, verbose=True):
    if "clip" in args.load_base_model:
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

    dataset = args.dataset.replace("_iid", "").split("_min")[0]
    args.embeddings_dir = paths.TARGET_CLASS_CACHE_DIR
    args.embeddings_dir = os.path.join(args.embeddings_dir, args.dataset)
    embedding_fname = f"d={dataset}-s={split}-m={args.load_base_model}.pt"
    embedding_path = os.path.join(args.embeddings_dir, embedding_fname)
    try:
        if os.path.exists(embedding_path):
            if verbose:
                print(f"-> Retrieving image embeddings from {embedding_path}!")
            embeddings = torch.load(embedding_path)

            return embeddings
        else:
            if verbose:
                print(f"-> Image embeddings from {embedding_path} not found.")

    except:
        pass

    model.to(args.device)
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for ix, data in enumerate(
            tqdm.tqdm(
                dataloader,
                desc=f"Computing {args.load_base_model} image embeddings for {split} split",
            )
        ):
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            embeddings = model.encode_image(inputs).float().cpu()
            all_embeddings.append(embeddings)
            inputs = inputs.cpu()
            labels = labels.cpu()
    model.cpu()

    torch.save(torch.cat(all_embeddings), embedding_path)
    if verbose:
        print(f"-> Saved image embeddings to {embedding_path}!")

    return torch.cat(all_embeddings)


def evaluate_clip(clip_predictions, dataloader, dataset_name, verbose=False):
    if dataset_name == "waterbirds" or dataset_name == "celebA":
        targets_t = dataloader.dataset.targets_all["target"].astype(int)
        targets_s = dataloader.dataset.targets_all["spurious"].astype(int)

        correct_by_groups = np.zeros(
            [len(np.unique(targets_t)), len(np.unique(targets_s))]
        )
        total_by_groups = np.zeros(correct_by_groups.shape)
        accs_by_group = np.zeros([len(np.unique(targets_t)), len(np.unique(targets_s))])

        correct = clip_predictions == targets_t

        for t in [0, 1]:
            for s in [0, 1]:
                ix = np.where(np.logical_and(targets_t == t, targets_s == s))[0]
                correct_by_groups[t][s] += np.sum(correct[ix])
                total_by_groups[t][s] += len(ix)
                accs_by_group[t][s] = np.sum(correct[ix]) / len(ix)

        avg_acc, robust_acc = summarize_acc(
            correct_by_groups, total_by_groups, stdout=verbose
        )

        if dataset_name == "waterbirds":
            adj_avg_acc = (
                accs_by_group[0][0] * 3498
                + accs_by_group[0][1] * 184
                + accs_by_group[1][0] * 56
                + accs_by_group[1][1] * 1057
            )
            adj_avg_acc = adj_avg_acc * 100 / (3498 + 184 + 56 + 1057)

        elif dataset_name == "celebA":
            adj_avg_acc = avg_acc

        gap = adj_avg_acc - robust_acc
        print("~ Target class results ~")
        print(f"WG : {robust_acc:.2f}%")
        print(f"Avg: {adj_avg_acc:.2f}%")
        print(f"Gap: {gap:.2f}%")
        print("------------------------------------")
        return avg_acc, robust_acc, gap, adj_avg_acc


def classify_with_embeddings(
    image_embeddings, text_embeddings, args, temperature=100.0
):
    with torch.no_grad():
        _image_embeddings = image_embeddings / image_embeddings.norm(
            dim=-1, keepdim=True
        )

        _text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

        cross = _image_embeddings @ _text_embeddings.T
        text_probs = (temperature * cross).softmax(dim=-1)
        _, predicted = torch.max(text_probs.data, 1)

    return predicted.cpu().numpy()


def get_zeroshot_predictions(key_embeddings, text_embeddings, args, temperature=100.0):
    predictions = classify_with_embeddings(
        key_embeddings, text_embeddings, args, temperature=100.0
    )

    return predictions
