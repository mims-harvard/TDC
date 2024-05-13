from .config import ResourceConfig
from ..feature_generators.resource import ResourceFeatureGenerator


class OpentargetsDTI(ResourceConfig):
    """Configuration for opentargets drug-target-identification datasets"""

    def __init__(self):
        super(OpentargetsDTI, self).__init__(
            ResourceFeatureGenerator(),
            keys=[
                "opentargets_ra_data_splits", "opentargets_ibd_data_splits",
                "df"
            ],
            loader_functions=["split", "split", "concat"],
            loader_args=[{
                "dataset": ("self", "opentargets_ra_drug_evidence"),
                "column_name":
                    "targetId_genename",
                "pos_train": ("self", [
                    "opentargets_ra_data_splits", "splits", "pos_train_names"
                ]),
                "pos_dev":
                    None,
                "pos_test":
                    ("self",
                     ["opentargets_ra_data_splits", "splits",
                      "pos_test_names"]),
                "neg_train": ("self", [
                    "opentargets_ra_data_splits", "splits", "neg_train_names"
                ]),
                "neg_dev":
                    None,
                "neg_test":
                    ("self",
                     ["opentargets_ra_data_splits", "splits",
                      "neg_test_names"]),
            }, {
                "dataset": ("self", "opentargets_ibd_drug_evidence"),
                "column_name":
                    "targetId_genename",
                "pos_train": ("self", [
                    "opentargets_ibd_data_splits", "splits", "pos_train_names"
                ]),
                "pos_dev":
                    None,
                "pos_test": ("self", [
                    "opentargets_ibd_data_splits", "splits", "pos_test_names"
                ]),
                "neg_train": ("self", [
                    "opentargets_ibd_data_splits", "splits", "neg_train_names"
                ]),
                "neg_dev":
                    None,
                "neg_test": ("self", [
                    "opentargets_ibd_data_splits", "splits", "neg_test_names"
                ])
            }, {
                "ds_list": [
                    "opentargets_ibd_data_splits", "opentargets_ra_data_splits"
                ],
                "axis": 0
            }])
