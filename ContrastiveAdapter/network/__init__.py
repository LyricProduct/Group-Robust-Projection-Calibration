import numpy as np
import torch

from network.adapter import MultiLayerAdapter

import network.clip as base_clip


def get_optimizer(model, args):
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    return optimizer


def get_adapter(num_classes, args, query_embeddings=None):
    if "adapter_mt" in args.train_method:
        model = MultiLayerAdapter(
            input_dim=args.base_model_dim,
            hidden_dim=args.hidden_dim,
            batch_norm=args.adapter_head_batch_norm,
            residual_connection=args.residual_connection,
            queries=query_embeddings,
            temperature=args.classification_temperature,
            return_hidden_by_default=False,
            num_encoder_layers=args.num_encoder_layers,
            activation=args.activation,
        )
    else:
        raise NotImplementedError
    return model


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
