import numpy as np
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from clip import clip
from datasets import initialize_data
from network import load_base_model
from network.clip import evaluate_clip
from utils import initialize_experiment
from sklearn.decomposition import TruncatedSVD
import warnings
import csv
import os
import pickle
from torchvision.datasets import (
    ImageNet,
    EuroSAT,
    Food101,
    Flowers102,
    Places365,
    OxfordIIITPet,
    DTD,
    FGVCAircraft,
)
import pathlib
import json
from datasets.datasets import CUBDataset, _transform
from tqdm import tqdm
from imagenetv2_pytorch import ImageNetV2Dataset as ImageNetV2
from configs import paths


def get_args():
    parser = argparse.ArgumentParser(description="Debiasing Vision-Language Models")

    parser.add_argument("--debias", default=False, action="store_true")
    parser.add_argument("--alpha", default=1000, type=float)

    parser.add_argument("--dataset", type=str, default="waterbirds")
    parser.add_argument("--num_workers", default=2, type=int)

    parser.add_argument("--load_base_model", type=str, default="")

    parser.add_argument("--bs_trn", default=128, type=int)
    parser.add_argument("--bs_val", default=128, type=int)

    parser.add_argument("--no_cuda", default=False, action="store_true")
    parser.add_argument("--verbose", default=False, action="store_true")

    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--retest_dataset", type=str, default="imagenet")
    parser.add_argument("--retest_batch_size", default=128, type=int)

    parser.add_argument("--beta", default=0, type=float)
    parser.add_argument("--gamma", default=1, type=float)
    parser.add_argument("--T_mode", type=str, default="None")
    parser.add_argument("--T_size", type=int, default=2500)

    args = parser.parse_args()

    args.arch = args.load_base_model.split("_")[-1].replace("/", "_").replace("-", "_")

    if torch.cuda.is_available() and args.no_cuda is False:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    return args


def get_proj_matrix(embeddings):
    tSVD = TruncatedSVD(n_components=len(embeddings))
    embeddings_ = tSVD.fit_transform(embeddings)
    basis = tSVD.components_.T

    proj = np.linalg.inv(np.matmul(basis.T, basis))
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    proj = np.eye(proj.shape[0]) - proj
    return proj


def get_A(z_i, z_j):
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return (
        np.matmul(z_i, z_i.T)
        + np.matmul(z_j, z_j.T)
        - np.matmul(z_i, z_j.T)
        - np.matmul(z_j, z_i.T)
    )


def get_M(embeddings, S):
    d = embeddings.shape[1]
    M = np.zeros((d, d))
    for s in S:
        M += get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)


def get_B(z_k):
    z_k = z_k[:, None]

    return np.matmul(z_k, z_k.T)


def get_imagenet_synsets_prompts(path):
    class_prompts = []

    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                class_prompts.append(line.strip())

    return class_prompts


def get_wiki_nouns_prompts(path):
    class_prompts = []

    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                class_prompts.append(line.strip())

    return class_prompts


def get_google_prompts(path):
    class_prompts = []

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                class_prompts.append(line.strip())
        return class_prompts

    return class_prompts


def print_info(args):
    print("------------------------------------")
    print(f"Model: {args.load_base_model.split('_')[1]}")
    print(f"Target Dataset: {args.dataset}")
    print(f"Non-target Dataset: {args.retest_dataset}")
    print(f"General Vocabulary: {args.T_mode}")
    print("------------------------------------")


def main():
    warnings.filterwarnings("ignore")
    args = get_args()
    print_info(args)

    base_model_args = args.load_base_model.split("_")

    base_model_components = load_base_model(base_model_args, args, clip=clip)
    (
        base_model,
        base_transform,
        get_embeddings,
        get_dataset_embeddings,
        get_zeroshot_predictions,
    ) = base_model_components

    load_dataloaders = initialize_data(args)
    dataloaders_base = load_dataloaders(
        args, train_shuffle=False, transform=base_transform
    )
    train_loader_base, val_loader_base, test_loader_base = dataloaders_base
    splits = ["train", "val", "test"]

    initialize_experiment(args)

    dataset_embeddings = {}
    for dix, split in enumerate(splits):
        dataset_embeddings[split] = get_dataset_embeddings(
            base_model, dataloaders_base[dix], args, split=split
        ).to(device=args.device)

    print(dataset_embeddings["train"].shape)
    args.base_model_dim = dataset_embeddings["train"].shape[1]
    print(f"-> Embedding dimensions: {args.base_model_dim}")
    print("------------------------------------")

    if args.dataset == "waterbirds":
        text_descriptions = [
            "This is a picture of a landbird.",
            "This is a picture of a waterbird.",
        ]
    elif args.dataset == "celebA":
        text_descriptions = [
            "A photo of a celebrity with dark hair.",
            "A photo of a celebrity with blond hair.",
        ]

    query_embeddings = get_embeddings(
        text_descriptions, base_model, args, normalize=True, verbose=True
    )

    if args.dataset == "waterbirds":
        spurious_prompt = [
            "This is a land background.",
            "This is a picture of a forest.",
            "This is a picture of a moutain.",
            "This is a picture of a wood.",
            "This is a water background.",
            "This is a picture of an ocean.",
            "This is a picture of a beach.",
            "This is a picture of a port.",
        ]
        candidate_prompt = [
            "This is a picture of a landbird with land background.",
            "This is a picture of a landbird with water background.",
            "This is a picture of a landbird in the ocean",
            "This is a picture of a landbird in the water.",
            "This is a picture of a landbird in the forest.",
            "This is a picture of a waterbird with land background.",
            "This is a picture of a waterbird with water background.",
            "This is a picture of a waterbird in the ocean",
            "This is a picture of a waterbird in the water.",
            "This is a picture of a waterbird in the forest.",
        ]
        S = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4],
            [5, 6],
            [5, 7],
            [5, 8],
            [5, 9],
            [6, 7],
            [6, 8],
            [6, 9],
            [7, 8],
            [7, 9],
            [8, 9],
        ]
    elif args.dataset == "celebA":
        spurious_prompt = [
            "A photo of a male.",
            "A photo of a male celebrity.",
            "A photo of a man.",
            "A photo of a female.",
            "A photo of a female celebrity.",
            "A photo of a woman.",
        ]

        candidate_prompt = [
            "A photo of a male celebrity with dark hair.",
            "A photo of a female celebrity with dark hair.",
            "A photo of a male celebrity with blond hair.",
            "A photo of a female celebrity with blond hair.",
        ]
        S = [[0, 1], [2, 3]]

    if args.debias:
        spurious_embeddings = get_embeddings(
            spurious_prompt, base_model, args, normalize=True, verbose=True
        )
        spurious_embeddings = spurious_embeddings.numpy()
        P0 = get_proj_matrix(spurious_embeddings)

        candidate_embeddings = get_embeddings(
            candidate_prompt, base_model, args, normalize=True, verbose=True
        )
        candidate_embeddings = candidate_embeddings.numpy()

        print("Solve Closed Form Optimum")
        M = get_M(candidate_embeddings, S)

    if args.debias:
        if args.T_mode == "wiki":
            wiki_nouns_namelist_path = "./general_vocabulary/Wiki_prompts.txt"
            retest_prompt = get_wiki_nouns_prompts(wiki_nouns_namelist_path)

            np.random.seed(args.seed)
            subset_indices = np.random.choice(
                len(retest_prompt), size=args.T_size, replace=False
            )
            subset = [retest_prompt[i] for i in subset_indices]
            retest_prompt = subset
        elif args.T_mode == "IN_syns":
            non_target_class_namelist_path = (
                "./general_vocabulary/ImageNet_Synsets_prompts.txt"
            )
            retest_prompt = get_imagenet_synsets_prompts(non_target_class_namelist_path)
        elif args.T_mode == "google":
            google_namelist_path = "./general_vocabulary/Google_prompts.txt"
            retest_prompt = get_google_prompts(google_namelist_path)

            np.random.seed(args.seed)
            subset_indices = np.random.choice(
                len(retest_prompt), size=args.T_size, replace=False
            )
            subset = [retest_prompt[i] for i in subset_indices]
            retest_prompt = subset

        if args.T_mode == "None":
            retest_embeddings = np.zeros((args.T_size, query_embeddings.shape[1]))
        else:
            retest_embeddings = get_embeddings(
                retest_prompt, base_model, args, normalize=True, verbose=False
            )
            retest_embeddings = retest_embeddings.numpy()

        G = np.eye(M.shape[0]) + args.alpha * M
        P = np.matmul(P0, np.linalg.inv(G))

        Z_c_T = query_embeddings.T
        Z_n_T = retest_embeddings.T

        X = np.concatenate([Z_c_T, Z_n_T], axis=1)
        T = np.concatenate([P @ Z_c_T.numpy(), Z_n_T], axis=1)

        c = Z_c_T.shape[1]
        n = Z_n_T.shape[1]

        W = np.diag([1.0] * c + [args.gamma] * n)

        comp1 = (T - P @ X) @ W
        comp2 = X @ W @ X.T

        Delta = comp1 @ X.T @ np.linalg.inv(comp2)

        if args.beta != 0:
            P = P + 1 / args.beta * Delta
        else:
            P = P

        text_embeddings = np.matmul(query_embeddings, P.T)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        text_embeddings = torch.tensor(text_embeddings).float()

    else:
        text_embeddings = query_embeddings
        text_embeddings = torch.tensor(text_embeddings).float()

    dataset_predictions = {}
    for dix, split in enumerate(splits):
        dataset_predictions[split] = get_zeroshot_predictions(
            dataset_embeddings[split],
            text_embeddings.to(device=args.device),
            args,
            temperature=100.0,
        )
    for ix, split in enumerate(splits):
        avg_acc, robust_acc, gap, adj_avg_acc = evaluate_clip(
            dataset_predictions[split],
            dataloaders_base[ix],
            args.dataset,
            verbose=True,
        )

    precomputed_encs_folder = paths.NON_TARGET_CLASS_CACHE_DIR
    os.makedirs(precomputed_encs_folder, exist_ok=True)
    precomputed_encs_file = os.path.join(
        precomputed_encs_folder,
        f"{args.retest_dataset}_{base_model_args[1]}.pkl",
    )

    if args.retest_dataset is not None:
        if args.retest_dataset == "imagenet":
            classnames_file_path = "./descriptors/imagenet_classnames.txt"

            args.retest_classes = []

            with open(classnames_file_path, "r") as f:
                for line in f:
                    class_name = line.strip().split(" ", 1)[1]
                    args.retest_classes.append(class_name)

            retest_prompt = [
                (f"A photo of an {c}." if c[0] in "aeiou" else f"A photo of a {c}.")
                for c in args.retest_classes
            ]
            args.retest_num_classes = len(args.retest_classes)
        elif args.retest_dataset == "eurosat":
            json_file_path = "./descriptors/descriptors_eurosat.json"

            with open(json_file_path, "r") as f:
                data = json.load(f)

            args.retest_classes = list(data.keys())

            retest_prompt = [
                (f"A photo of an {c}." if c[0] in "aeiou" else f"A photo of a {c}.")
                for c in args.retest_classes
            ]
            args.retest_num_classes = len(args.retest_classes)
        elif args.retest_dataset == "cub":
            json_file_path = "./descriptors/descriptors_cub.json"

            with open(json_file_path, "r") as f:
                data = json.load(f)

            args.retest_classes = list(data.keys())

            retest_prompt = [
                (f"A photo of an {c}." if c[0] in "aeiou" else f"A photo of a {c}.")
                for c in args.retest_classes
            ]
            args.retest_num_classes = len(args.retest_classes)
        elif args.retest_dataset == "food101":
            json_file_path = "./descriptors/descriptors_food101.json"

            with open(json_file_path, "r") as f:
                data = json.load(f)

            args.retest_classes = list(data.keys())

            retest_prompt = [
                (f"A photo of an {c}." if c[0] in "aeiou" else f"A photo of a {c}.")
                for c in args.retest_classes
            ]
            args.retest_num_classes = len(args.retest_classes)
        elif args.retest_dataset == "imagenetv2":
            classnames_file_path = "./descriptors/imagenet_classnames.txt"

            args.retest_classes = []

            with open(classnames_file_path, "r") as f:
                for line in f:
                    class_name = line.strip().split(" ", 1)[1]
                    args.retest_classes.append(class_name)

            retest_prompt = [
                (f"A photo of an {c}." if c[0] in "aeiou" else f"A photo of a {c}.")
                for c in args.retest_classes
            ]
            args.retest_num_classes = len(args.retest_classes)
        elif args.retest_dataset == "places365":
            json_file_path = "./descriptors/descriptors_places365.json"

            with open(json_file_path, "r") as f:
                data = json.load(f)

            args.retest_classes = list(data.keys())

            retest_prompt = [
                (f"A photo of an {c}." if c[0] in "aeiou" else f"A photo of a {c}.")
                for c in args.retest_classes
            ]
            args.retest_num_classes = len(args.retest_classes)
        elif args.retest_dataset == "pets":
            json_file_path = "./descriptors/descriptors_pets.json"

            with open(json_file_path, "r") as f:
                data = json.load(f)

            args.retest_classes = list(data.keys())

            retest_prompt = {
                c: (f"A photo of an {c}." if c[0] in "aeiou" else f"A photo of a {c}.")
                for c in args.retest_classes
            }

            sorted_keys = np.array(list(retest_prompt.keys()))[
                np.argsort([x.lower() for x in retest_prompt.keys()])
            ]
            retest_prompt = {key: retest_prompt[key] for key in sorted_keys}
            retest_prompt = list(retest_prompt.values())

            args.retest_num_classes = len(args.retest_classes)
        elif args.retest_dataset == "dtd":
            json_file_path = "./descriptors/descriptors_dtd.json"

            with open(json_file_path, "r") as f:
                data = json.load(f)

            args.retest_classes = list(data.keys())

            retest_prompt = [
                (f"A photo of an {c}." if c[0] in "aeiou" else f"A photo of a {c}.")
                for c in args.retest_classes
            ]
            args.retest_num_classes = len(args.retest_classes)
        elif args.retest_dataset == "flowers102":
            json_file_path = "./descriptors/descriptors_flowers102.json"

            with open(json_file_path, "r") as f:
                data = json.load(f)

            args.retest_classes = list(data.keys())

            retest_prompt = [
                (f"A photo of an {c}." if c[0] in "aeiou" else f"A photo of a {c}.")
                for c in args.retest_classes
            ]
            args.retest_num_classes = len(args.retest_classes)
        elif args.retest_dataset == "fgvcaircraft":
            json_file_path = "./descriptors/descriptors_fgvcaircraft.json"

            with open(json_file_path, "r") as f:
                data = json.load(f)

            args.retest_classes = list(data.keys())

            retest_prompt = [
                (f"A photo of an {c}." if c[0] in "aeiou" else f"A photo of a {c}.")
                for c in args.retest_classes
            ]
            args.retest_num_classes = len(args.retest_classes)
        elif args.retest_dataset == "cars":
            json_file_path = "./descriptors/descriptors_cars.json"

            with open(json_file_path, "r") as f:
                data = json.load(f)

            args.retest_classes = list(data.keys())

            retest_prompt = [
                (f"A photo of an {c}." if c[0] in "aeiou" else f"A photo of a {c}.")
                for c in args.retest_classes
            ]
            args.retest_num_classes = len(args.retest_classes)

    if os.path.exists(precomputed_encs_file):
        load_res = pickle.load(open(precomputed_encs_file, "rb"))
    else:
        args.image_size = 224

        args.tfms = _transform(args.image_size)

        if args.retest_dataset == "imagenet":
            dsclass = ImageNet
            args.data_dir = pathlib.Path(paths.IMAGENET_DIR)
            dataset = dsclass(args.data_dir, split="val", transform=args.tfms)
            args.classes_to_load = None
            args.descriptor_fname = "descriptors_imagenet"
        elif args.retest_dataset == "eurosat":
            dsclass = EuroSAT
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            args.data_dir = pathlib.Path(paths.EUROSAT_DIR)
            dataset = dsclass(args.data_dir, transform=args.tfms, download=True)
            args.classes_to_load = None
            args.descriptor_fname = "descriptors_eurosat"
        elif args.retest_dataset == "cub":

            args.data_dir = pathlib.Path(paths.CUB_DIR)
            dataset = CUBDataset(args.data_dir, train=False, transform=args.tfms)
            args.classes_to_load = None
            args.descriptor_fname = "descriptors_cub"
        elif args.retest_dataset == "food101":
            dsclass = Food101
            args.data_dir = pathlib.Path(paths.FOOD101_DIR)
            dataset = dsclass(
                args.data_dir, split="test", transform=args.tfms, download=True
            )
            args.classes_to_load = None
            args.descriptor_fname = "descriptors_food101"
        elif args.retest_dataset == "imagenetv2":
            args.data_dir = pathlib.Path(paths.IMAGENETV2_DIR)
            dataset = ImageNetV2(location=args.data_dir, transform=args.tfms)

            args.classes_to_load = None
            args.descriptor_fname = "descriptors_imagenet"
            dataset.classes = args.classes_to_load
        elif args.retest_dataset == "places365":
            dsclass = Places365
            args.data_dir = pathlib.Path(paths.PLACES365_DIR)
            download = True
            if os.path.exists(os.path.join(args.data_dir, "val_256")):
                download = False
            dataset = dsclass(
                args.data_dir,
                split="val",
                small=True,
                transform=args.tfms,
                download=download,
            )
            args.classes_to_load = None
            args.descriptor_fname = "descriptors_places365"
        elif args.retest_dataset == "pets":
            dsclass = OxfordIIITPet
            args.data_dir = pathlib.Path(paths.PETS_DIR)
            dataset = dsclass(
                args.data_dir, split="test", transform=args.tfms, download=True
            )
            args.classes_to_load = None
            args.descriptor_fname = "descriptors_pets"
        elif args.retest_dataset == "dtd":
            dsclass = DTD
            args.data_dir = pathlib.Path(paths.DTD_DIR)
            dataset = dsclass(
                args.data_dir, split="test", transform=args.tfms, download=True
            )
            args.classes_to_load = None
            args.descriptor_fname = "descriptors_dtd"
        elif args.retest_dataset == "flowers102":
            dsclass = Flowers102
            args.data_dir = pathlib.Path(paths.FLOWERS102_DIR)
            dataset = dsclass(
                args.data_dir, split="test", transform=args.tfms, download=True
            )
            args.classes_to_load = None
            args.descriptor_fname = "descriptors_flowers102"
        elif args.retest_dataset == "fgvcaircraft":
            dsclass = FGVCAircraft
            args.data_dir = pathlib.Path(paths.FGVCAIRCRAFT_DIR)
            dataset = dsclass(
                args.data_dir,
                split="test",
                annotation_level="family",
                transform=args.tfms,
                download=True,
            )
            args.classes_to_load = None
            args.descriptor_fname = "descriptors_fgvcaircraft"
        elif args.retest_dataset == "cars":
            from scipy.io import loadmat
            from PIL import Image

            class StanfordCars(torch.utils.data.Dataset):

                def __init__(self, root_path, transform=None):
                    self.root_path = pathlib.Path(os.path.join(root_path, "cars_test"))
                    file_path = os.path.join(
                        root_path, "cars_test_annos_withlabels_renew"
                    )
                    root_path = self.root_path
                    self.images = [
                        os.path.join(root_path, file)
                        for file in sorted(os.listdir(self.root_path))
                    ]
                    self.transform = transform

                    annos = loadmat(file_path)
                    annotations = annos["annotations"]
                    labels = annotations["class"]
                    labels_list = [int(label[0][0]) - 1 for label in labels[0]]
                    self.labels = labels_list

                def __len__(self):
                    return len(self.images)

                def __getitem__(self, index):
                    image_file = self.images[index]
                    image = Image.open(image_file).convert("RGB")
                    if self.transform:
                        image = self.transform(image)

                    return image, self.labels[index]

            dsclass = StanfordCars
            args.data_dir = pathlib.Path(paths.CARS_DIR)
            dataset = dsclass(args.data_dir, transform=args.tfms)
            args.classes_to_load = None
            args.descriptor_fname = "descriptors_cars"

        dataloader = DataLoader(
            dataset,
            args.retest_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        enc_coll = []
        label_coll = []

        base_model = base_model.to(args.device)
        base_model.eval()
        base_model.requires_grad_(False)

        with torch.no_grad():
            for batch_number, batch in enumerate(
                tqdm(dataloader, desc="Precomputing image embeddings...")
            ):
                images, labels = batch
                images = images.to(args.device)
                labels = labels.to(args.device)
                label_coll.append(labels)

                image_encodings = F.normalize(base_model.encode_image(images.half()))
                enc_coll.append(image_encodings.cpu())
        load_res = {"enc": enc_coll, "labels": label_coll}
        pickle.dump(load_res, open(precomputed_encs_file, "wb"))

    encoding_coll = load_res["enc"]
    label_coll = load_res["labels"]

    query_embeddings = get_embeddings(
        retest_prompt, base_model, args, normalize=True, verbose=False
    )
    text_embeddings = query_embeddings.float().to(args.device)

    correct = 0
    total = 0

    if not args.debias:
        P = np.eye(query_embeddings.shape[1])

    text_embeddings = np.matmul(query_embeddings, P.T)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    text_embeddings = torch.tensor(text_embeddings).float()

    for batch_number, (image_encodings, labels) in tqdm(
        enumerate(zip(encoding_coll, label_coll)),
        total=len(encoding_coll),
        desc="Evaluating non-target classes",
    ):
        image_encodings = image_encodings.to(args.device)
        labels = labels.to(args.device)

        retest_predictions = get_zeroshot_predictions(
            image_encodings.to(torch.float32),
            text_embeddings.to(args.device),
            args,
            temperature=100.0,
        )

        if isinstance(retest_predictions, np.ndarray):
            retest_predictions = torch.from_numpy(retest_predictions).to(args.device)

        retest_predictions = retest_predictions.float()

        correct += (retest_predictions == labels).sum().item()
        total += labels.size(0)

    acc_cali = correct / total * 100
    acc_cali = round(acc_cali, 1)
    print(f"Non-target Accuracy: {acc_cali} %")
    print("\n\n")

    robust_acc = round(robust_acc, 1)
    adj_avg_acc = round(adj_avg_acc, 1)
    gap = round(gap, 1)
    acc_cali = round(acc_cali, 1)

    result_dir_path = f"./Result/{args.dataset}/"
    os.makedirs(result_dir_path, exist_ok=True)

    csv_file_path = f"{result_dir_path}/result_{args.T_mode}.csv"

    file_exists = os.path.isfile(csv_file_path)

    with open(csv_file_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                ["target classes", "WG", "Avg", "Gap", "non-target classes", "Acc"]
            )

        writer.writerow(
            [
                args.dataset,
                robust_acc,
                adj_avg_acc,
                gap,
                args.retest_dataset,
                acc_cali,
            ]
        )


if __name__ == "__main__":
    main()
