import os
import torch
import pandas as pd
from PIL import Image
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.metric import ElementwiseMetric
from wilds.common.utils import minimum


class Accuracy(ElementwiseMetric):
    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn
        if name is None:
            name = "acc"
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        if self.prediction_fn is not None:
            y_pred = self.prediction_fn(y_pred)
        return (y_pred == y_true).float()

    def worst(self, metrics):
        return minimum(metrics)


class CelebADataset(WILDSDataset):

    _dataset_name = "celebA"
    _versions_dict = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0xfe55077f5cd541f985ebf9ec50473293/contents/blob/",
            "compressed_size": 1_308_557_312,
        }
    }

    def __init__(
        self,
        version=None,
        root_dir=None,
        download=False,
        split_scheme="official",
    ):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        target_name = "Blond_Hair"
        confounder_names = ["Male"]

        attrs_df = pd.read_csv(os.path.join(self.data_dir, "list_attr_celeba.csv"))

        self._input_array = attrs_df["image_id"].values
        self._original_resolution = (178, 218)
        attrs_df = attrs_df.drop(labels="image_id", axis="columns")
        attr_names = attrs_df.columns.copy()

        def attr_idx(attr_name):
            return attr_names.get_loc(attr_name)

        attrs_df = attrs_df.values
        attrs_df[attrs_df == -1] = 0

        target_idx = attr_idx(target_name)
        self._y_array = torch.LongTensor(attrs_df[:, target_idx])
        self._y_size = 1
        self._n_classes = 2

        confounder_idx = [attr_idx(a) for a in confounder_names]
        confounders = attrs_df[:, confounder_idx]

        self._metadata_array = torch.cat(
            (torch.LongTensor(confounders), self._y_array.reshape((-1, 1))), dim=1
        )
        confounder_names = [s.lower() for s in confounder_names]
        self._metadata_fields = confounder_names + ["y"]
        self._metadata_map = {"y": ["not blond", "    blond"]}

        self._eval_grouper = CombinatorialGrouper(
            dataset=self, groupby_fields=(confounder_names + ["y"])
        )

        self._split_scheme = split_scheme
        if self._split_scheme != "official":
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")
        split_df = pd.read_csv(os.path.join(self.data_dir, "list_eval_partition.csv"))
        self._split_array = split_df["partition"].values

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        img_filename = os.path.join(
            self.data_dir, "img_align_celeba", self._input_array[idx]
        )
        x = Image.open(img_filename).convert("RGB")
        return x

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric, self._eval_grouper, y_pred, y_true, metadata
        )
