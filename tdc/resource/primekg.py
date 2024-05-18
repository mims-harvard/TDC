# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT
"""
This file contains a primekg dataloader. 
"""

import numpy as np
import warnings

from ..utils import general_load
from ..utils.knowledge_graph import KnowledgeGraph

warnings.filterwarnings("ignore")


class PrimeKG(KnowledgeGraph):
    """PrimeKG data loader class to load the knowledge graph with additional support functions.
    """

    def __init__(self, path="./data"):
        """load the KG to the specified path"""
        df = general_load("primekg", path, ",")
        self.df = df
        self.path = path
        super().__init__(self.df)

    def get_data(self):
        return self.df

    def to_nx(self):
        import networkx as nx

        G = nx.Graph()
        for i in self.df.relation.unique():
            G.add_edges_from(self.df[self.df.relation == i][["x_id",
                                                             "y_id"]].values,
                             relation=i)
        return G

    def get_features(self, feature_type):
        if feature_type not in ["drug", "disease"]:
            raise ValueError("feature_type only supports drug/disease!")
        return general_load("primekg_" + feature_type + "_feature", self.path,
                            "\t")

    def get_node_list(self, node_type):
        df = self.df
        return np.unique(df[(df.x_type == node_type)].x_id.unique().tolist() +
                         df[(df.y_type == node_type)].y_id.unique().tolist())
