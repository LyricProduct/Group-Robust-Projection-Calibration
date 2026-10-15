import numpy as np

import network.clip as base_clip


def get_parameter_count(model, verbose=True):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    if verbose:
        print(f"-> Number of parameters: {num_params}")
    return num_params


def load_base_model(base_model_args, args, clip=None):
    clip_name = base_model_args[1]
    base_model, transform = clip.load(clip_name)
    args.num_base_model_parameters = get_parameter_count(base_model, verbose=True)
    base_transform = transform
    get_embeddings = base_clip.get_embeddings
    get_dataset_embeddings = base_clip.get_dataset_embeddings
    get_zeroshot_predictions = base_clip.get_zeroshot_predictions

    return (
        base_model,
        base_transform,
        get_embeddings,
        get_dataset_embeddings,
        get_zeroshot_predictions,
    )
