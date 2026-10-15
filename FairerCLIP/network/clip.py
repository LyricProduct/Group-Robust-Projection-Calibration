import numpy as np
import torch
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


def evaluate_clip(clip_predictions, target, group, verbose=False):
    targets_t = target.cpu().numpy().astype(int).argmax(1)
    targets_s = group.cpu().numpy().astype(int).argmax(1)

    correct_by_groups = np.zeros([len(np.unique(targets_t)), len(np.unique(targets_s))])
    total_by_groups = np.zeros(correct_by_groups.shape)

    correct = clip_predictions == targets_t
    for ix, y in enumerate(targets_t):
        s = targets_s[ix]
        correct_by_groups[int(y)][int(s)] += correct[ix].item()
        total_by_groups[int(y)][int(s)] += 1

    avg_acc, robust_acc, groups_acc = summarize_acc(
        correct_by_groups, total_by_groups, return_groups=True, stdout=verbose
    )

    return avg_acc, robust_acc, groups_acc


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
        key_embeddings, text_embeddings, args, temperature=temperature
    )
    return predictions
