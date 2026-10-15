import torch
from torch.utils.data import Dataset, DataLoader


def generate_random_list(n, m, seed=42):
    torch.manual_seed(seed)
    random_list = torch.randint(0, m, (n,))
    return random_list.tolist()


class Bird_SSL_Dataset(Dataset):
    def __init__(self, birds, backgrounds, num_samples=100, seed=42):
        super().__init__()
        self.num_samples = num_samples
        self.birds1 = birds[0]
        self.birds2 = birds[1]

        self.backgrounds1 = backgrounds[0]
        self.backgrounds2 = backgrounds[1]

        self.birds1_idx = generate_random_list(num_samples, len(self.birds1), seed=seed)
        self.birds2_idx = generate_random_list(num_samples, len(self.birds2), seed=seed)
        self.backgrounds1_idx = generate_random_list(
            num_samples, len(self.backgrounds1), seed=seed
        )
        self.backgrounds2_idx = generate_random_list(
            num_samples, len(self.backgrounds2), seed=seed
        )
        self.bird1vsbird2 = generate_random_list(num_samples, 2, seed=seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        bird1_idx = self.birds1_idx[idx]
        bird2_idx = self.birds2_idx[idx]
        background1_idx = self.backgrounds1_idx[idx]
        background2_idx = self.backgrounds2_idx[idx]

        bird1 = self.birds1[bird1_idx]
        bird2 = self.birds2[bird2_idx]

        background1 = self.backgrounds1[background1_idx]
        background2 = self.backgrounds2[background2_idx]

        sentence_neg_1 = f"A photo of a {bird1}"
        sentence_neg_2 = f"A photo of a {bird2}"

        if self.bird1vsbird2[idx] == 1:
            pos_bird = bird1
        else:
            pos_bird = bird2

        sentence_pos_1 = f"A photo of a {pos_bird} in a {background1}."
        sentence_pos_2 = f"A photo of a {pos_bird} in a {background2}."

        sentences = [sentence_neg_1, sentence_neg_2, sentence_pos_1, sentence_pos_2]

        return sentences


class Celeb_SSL_Dataset(Dataset):
    def __init__(self, args, scene_descriptions, genders, blond, dark, num_samples=100):
        super().__init__()
        self.num_samples = num_samples

        self.sd = scene_descriptions
        self.gender = genders
        self.hair1 = blond
        self.hair2 = dark

        self.sd_idx = generate_random_list(num_samples, len(self.sd), seed=args.seed)
        self.gender_idx = generate_random_list(
            num_samples, len(self.gender), seed=args.seed
        )
        self.hair1_idx = generate_random_list(
            num_samples, len(self.hair1), seed=args.seed
        )
        self.hair2_idx = generate_random_list(
            num_samples, len(self.hair2), seed=args.seed
        )
        self.hair1vshair2 = generate_random_list(num_samples, 2)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        scene_idx = self.sd_idx[idx]
        gender_idx = self.gender_idx[idx]
        hair1_idx = self.hair1_idx[idx]
        hair2_idx = self.hair2_idx[idx]

        sentence_neg_1 = (
            self.sd[scene_idx]
            .replace("*Gender*", self.gender[gender_idx])
            .replace("*hair*", self.hair1[hair1_idx])
        )
        sentence_neg_2 = (
            self.sd[scene_idx]
            .replace("*Gender*", self.gender[gender_idx])
            .replace("*hair*", self.hair2[hair2_idx])
        )

        if self.hair1vshair2[idx] == 1:
            pos_hair = self.hair1[hair1_idx]
        else:
            pos_hair = self.hair2[hair2_idx]

        if gender_idx == 0:
            opp_gender = 1
        else:
            opp_gender = 0
        sentence_pos_1 = (
            self.sd[scene_idx]
            .replace("*Gender*", self.gender[gender_idx])
            .replace("*hair*", pos_hair)
        )
        sentence_pos_2 = (
            self.sd[scene_idx]
            .replace("*Gender*", self.gender[opp_gender])
            .replace("*hair*", pos_hair)
        )

        sentences = [sentence_neg_1, sentence_neg_2, sentence_pos_1, sentence_pos_2]

        return sentences


def read_lines(path):
    with open(path, "r") as file:
        return [line.strip() for line in file]


def get_SSL_dataset(args):
    if args.dataset == "celeba":
        textset = Celeb_SSL_Dataset(
            args,
            read_lines("data/scene_descriptions.txt"),
            read_lines("data/genders.txt"),
            read_lines("data/blond_hair_descriptions.txt"),
            read_lines("data/dark_hair_descriptions.txt"),
            num_samples=args.num_samples,
        )
    elif args.dataset == "waterbirds":
        birds = [read_lines("data/land_birds.txt"), read_lines("data/water_birds.txt")]
        backgrounds = [
            read_lines("data/water_backgrounds.txt"),
            read_lines("data/ground_backgrounds.txt"),
        ]
        textset = Bird_SSL_Dataset(
            birds, backgrounds, num_samples=args.num_samples, seed=args.seed
        )
    textloader = DataLoader(textset, batch_size=3, shuffle=True)

    return textloader
