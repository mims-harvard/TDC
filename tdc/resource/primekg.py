# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT
"""
This file contains a primekg dataloader. 
"""

import pandas as pd
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore")

from ..utils import general_load


class PrimeKG:

    """PrimeKG data loader class to load the knowledge graph with additional support functions."""

    def __init__(self, path="./data"):
        """load the KG to the specified path"""
        self.df = general_load("primekg", path, ",")
        self.path = path

    def get_data(self):
        return self.df

    def to_nx(self):
        import networkx as nx

        G = nx.Graph()
        for i in self.df.relation.unique():
            G.add_edges_from(
                self.df[self.df.relation == i][["x_id", "y_id"]].values, relation=i
            )
        return G

    def get_nodes_by_source(self, source):
        out = pd.concat([
            self.df.query(
                f"x_source == '{source}' | y_source == '{source}'"
            )[['x_index', 'x_id', 'x_type', 'x_name', 'x_source']].rename(
                columns={'x_index': 'index', 'x_id': 'id', 'x_type': 'type', 'x_name': 'name',
                         'x_source': 'source'}),

            self.df.query(
                f"x_source == '{source}' | y_source == '{source}'"
            )[['y_index', 'y_id', 'y_type', 'y_name', 'y_source']].rename(
                columns={'y_index': 'index', 'y_id': 'id', 'y_type': 'type', 'y_name': 'name',
                         'y_source': 'source'})
        ]).query(f'source == "{source}"').drop_duplicates().reset_index(drop=True).sort_values('index')

        return out

    def get_features(self, feature_type):
        if feature_type not in ["drug", "disease"]:
            raise ValueError("feature_type only supports drug/disease!")
        return general_load("primekg_" + feature_type + "_feature", self.path, "\t")

    def get_node_list(self, node_type):
        df = self.df
        return np.unique(
            df[(df.x_type == node_type)].x_id.unique().tolist()
            + df[(df.y_type == node_type)].y_id.unique().tolist()
        )
