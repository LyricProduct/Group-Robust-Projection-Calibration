import os
from os.path import join
from configs import paths


def initialize_save_paths(args):
    args.embeddings_dir = paths.TARGET_CLASS_CACHE_DIR
    args.embeddings_dir_target = join(args.embeddings_dir, args.dataset)
    os.makedirs(args.embeddings_dir_target, exist_ok=True)


def initialize_experiment(args):
    initialize_save_paths(args)

    print("---")
