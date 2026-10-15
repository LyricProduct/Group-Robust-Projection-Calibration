import network.clip as base_clip


def load_base_model(base_model_args, args, clip):
    clip_name = base_model_args[1]
    base_model, transform = clip.load(clip_name)
    return (
        base_model,
        transform,
        base_clip.get_embeddings,
        base_clip.get_zeroshot_predictions,
    )
