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


class WaterbirdsDataset(WILDSDataset):

    _dataset_name = "waterbirds"
    _versions_dict = {
        "1.0": {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/",
            "compressed_size": None,
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

        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"{self.data_dir} does not exist yet. Please generate the dataset first."
            )

        metadata_df = pd.read_csv(os.path.join(self.data_dir, "metadata.csv"))

        self._y_array = torch.LongTensor(metadata_df["y"].values)
        self._y_size = 1
        self._n_classes = 2

        self._metadata_array = torch.stack(
            (torch.LongTensor(metadata_df["place"].values), self._y_array), dim=1
        )
        self._metadata_fields = ["background", "y"]
        self._metadata_map = {
            "background": [" land", "water"],
            "y": [" landbird", "waterbird"],
        }

        self._input_array = metadata_df["img_filename"].values
        self._original_resolution = (224, 224)

        self._split_scheme = split_scheme
        if self._split_scheme != "official":
            raise ValueError(f"Split scheme {self._split_scheme} not recognized")
        self._split_array = metadata_df["split"].values

        self._eval_grouper = CombinatorialGrouper(
            dataset=self, groupby_fields=(["background", "y"])
        )

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        img_filename = os.path.join(self.data_dir, self._input_array[idx])
        x = Image.open(img_filename).convert("RGB")
        return x

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = Accuracy(prediction_fn=prediction_fn)

        results, results_str = self.standard_group_eval(
            metric, self._eval_grouper, y_pred, y_true, metadata
        )

        results["adj_acc_avg"] = (
            results["acc_y:landbird_background:land"] * 3498
            + results["acc_y:landbird_background:water"] * 184
            + results["acc_y:waterbird_background:land"] * 56
            + results["acc_y:waterbird_background:water"] * 1057
        ) / (3498 + 184 + 56 + 1057)

        del results["acc_avg"]
        results_str = (
            f"Adjusted average acc: {results['adj_acc_avg']:.3f}\n"
            + "\n".join(results_str.split("\n")[1:])
        )

        return results, results_str
