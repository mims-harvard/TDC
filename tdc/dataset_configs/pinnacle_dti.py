from .config import ResourceConfig
from ..feature_generators.resource import ResourceFeatureGenerator


class PinnacleDTI(ResourceConfig):
    """Configuration for PINNACLE drug-target-identification datasets"""

    def __init__(self):
        super(PinnacleDTI, self).__init__(
            ResourceFeatureGenerator(),
            keys=["pinnacle_ra_data_splits", "pinnacle_ibd_data_splits", "df"],
            loader_functions=["split", "split", "concat"],
            loader_args=[{
                "dataset": ("self","pinnacle_ra_drug_evidence"),
                "column_name": "targetId_genename",
                "pos_train": ("self", ["pinnacle_ra_data_splits", "splits", "pos_train_names"]),
                "pos_dev": None,
                "pos_test": ("self", ["pinnacle_ra_data_splits", "splits", "pos_test_names"]),
                "neg_train": ("self", ["pinnacle_ra_data_splits", "splits", "neg_train_names"]),
                "neg_dev": None,
                "neg_test": ("self", ["pinnacle_ra_data_splits", "splits", "neg_test_names"]),
            },
            {
                "dataset": ("self","pinnacle_ibd_drug_evidence"),
                "column_name": "targetId_genename",
                "pos_train": ("self", ["pinnacle_ibd_data_splits", "splits", "pos_train_names"]),
                "pos_dev": None,
                "pos_test": ("self", ["pinnacle_ibd_data_splits", "splits", "pos_test_names"]),
                "neg_train": ("self", ["pinnacle_ibd_data_splits", "splits", "neg_train_names"]),
                "neg_dev": None,
                "neg_test": ("self", ["pinnacle_ibd_data_splits", "splits", "neg_test_names"])
            },
            {
                "ds_list": ["pinnacle_ibd_data_splits", "pinnacle_ra_data_splits"],
                "axis": 0
            }
        ]
        )