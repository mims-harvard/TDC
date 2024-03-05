# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import sys
import warnings

warnings.filterwarnings("ignore")

from . import single_pred_dataset
from ..utils import print_sys
from ..metadata import dataset_names


class Develop(single_pred_dataset.DataLoader):
    """Data loader class to load datasets in Develop task. More info: https://tdcommons.ai/single_pred_tasks/develop/

    Args:
        name (str): the dataset name.
        path (str, optional):
            The path to save the data file, defaults to './data'
        label_name (str, optional):
            For multi-label dataset, specify the label name, defaults to None
        print_stats (bool, optional):
            Whether to print basic statistics of the dataset, defaults to False
        convert_format (str, optional):
            Automatic conversion of SMILES to other molecular formats in MolConvert class. Stored as separate column in dataframe, defaults to None
    """

    def __init__(
        self,
        name,
        path="./data",
        label_name=None,
        print_stats=False,
        convert_format=None,
    ):
        """Create a Developability prediction dataloader object."""
        super().__init__(
            name,
            path,
            label_name,
            print_stats,
            dataset_names=dataset_names["Develop"],
            convert_format=convert_format,
        )
        self.entity1_name = "Antibody"
        if print_stats:
            self.print_stats()
        print("Done!", flush=True, file=sys.stderr)

    def graphein(
        self,
        graph="distance",
        node_feature=["amino_acid_one_hot"],
        distance_threshold=6,
        config=None,
        convertor=None,
    ):
        from typing import Dict
        import torch
        from functools import partial

        import graphein.protein as gp
        from graphein.ml.conversion import GraphFormatConvertor
        from graphein.ml import InMemoryProteinGraphDataset, ProteinGraphDataset

        from graphein.protein.utils import get_obsolete_mapping
        import warnings

        warnings.filterwarnings("ignore")
        import logging

        logging.getLogger("graphein").setLevel("ERROR")
        try:
            split = self.split
        except:
            raise ValueError("Please define data.split() first!")

        print_sys(
            "Note that this task has obsolete PDBs, thus, we remove it in default. For fair comparison using other models, please retrieve the data.split object for the post removal splits!"
        )

        from graphein.protein.utils import get_obsolete_mapping

        obs = get_obsolete_mapping()
        train_obs = [
            t for t in split["train"]["Antibody_ID"] if t in obs.keys()
        ]
        valid_obs = [
            t for t in split["valid"]["Antibody_ID"] if t in obs.keys()
        ]
        test_obs = [t for t in split["test"]["Antibody_ID"] if t in obs.keys()]

        split["train"] = split["train"].loc[~split["train"]["Antibody_ID"].
                                            isin(train_obs)]
        split["test"] = split["test"].loc[~split["test"]["Antibody_ID"].
                                          isin(test_obs)]
        split["valid"] = split["valid"].loc[~split["valid"]["Antibody_ID"].
                                            isin(valid_obs)]

        self.split = split

        def get_label_map(split_name: str) -> Dict[str, torch.Tensor]:
            return dict(
                zip(
                    split[split_name].Antibody_ID,
                    split[split_name].Y.apply(torch.tensor),
                ))

        train_labels = get_label_map("train")
        valid_labels = get_label_map("valid")
        test_labels = get_label_map("test")

        if config is None:
            if graph == "distance":
                edge_fct = [
                    partial(
                        gp.add_distance_threshold,
                        threshold=distance_threshold,
                        long_interaction_threshold=0,
                    )
                ]

            node_feature_fct = []
            for i in node_feature:
                if i == "amino_acid_one_hot":
                    node_feature_fct.append(gp.amino_acid_one_hot)

            from functools import partial

            graphein_config = gp.ProteinGraphConfig(
                node_metadata_functions=node_feature_fct,
                edge_construction_functions=edge_fct,
            )

            convertor = GraphFormatConvertor(
                src_format="nx",
                dst_format="pyg",
                columns=["coords", "edge_index"] + node_feature,
            )
        else:
            graphein_config = config
            convertor = convertor

        train_ds = InMemoryProteinGraphDataset(
            root="./data/",
            name="train",
            pdb_codes=split["train"]["Antibody_ID"],
            graph_label_map=train_labels,
            graphein_config=graphein_config,
            graph_format_convertor=convertor,
        )

        valid_ds = InMemoryProteinGraphDataset(
            root="./data/",
            name="valid",
            pdb_codes=split["valid"]["Antibody_ID"],
            graph_label_map=valid_labels,
            graphein_config=graphein_config,
            graph_format_convertor=convertor,
        )

        test_ds = InMemoryProteinGraphDataset(
            root="./data/",
            name="test",
            pdb_codes=split["test"]["Antibody_ID"],
            graph_label_map=test_labels,
            graphein_config=graphein_config,
            graph_format_convertor=convertor,
        )

        return {"train": train_ds, "valid": valid_ds, "test": test_ds}
