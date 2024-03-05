"""A python module to build, handle, explore, and manipulate knowledge graphs.
"""

import pandas as pd
from copy import copy

kg_columns = [
    'relation', 'display_relation', 'x_id', 'x_type', 'x_name', 'x_source',
    'y_id', 'y_type', 'y_name', 'y_source'
]


class KnowledgeGraph:

    def __init__(self, df=None):
        if df is not None:
            self.df = df
        else:
            self.df = pd.DataFrame('', columns=kg_columns)

    def copy(self):
        return copy(self)

    def run_query(self, query):
        """build subgraph using given query"""
        self.df_raw = self.df
        self.df = self.df.query(query).reset_index(drop=True)

    def get_nodes_by_source(self, source):
        # extract x nodes
        x_df = self.df.query(
            f"x_source == '{source}' | y_source == '{source}'")[[
                col for col in self.df.columns if col.startswith("x_")
            ]]