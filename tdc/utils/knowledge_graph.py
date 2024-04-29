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

    def run_query(self, query, inplace=True):
        """build subgraph using given query"""
        df_filt = self.df.query(query).reset_index(drop=True)
        if inplace:
            self.df_raw = self.df
            self.df = df_filt
        else:
            return df_filt

    def get_nodes_by_source(self, source):
        # extract x nodes
        x_df = self.df.query(
            f"x_source == '{source}' | y_source == '{source}'")[[
                col for col in self.df.columns if col.startswith("x_")
            ]]

        for col in x_df.columns:
            x_df = x_df.rename(columns={col: col[2:]})

        # extract y nodes
        y_df = self.df.query(
            f"x_source == '{source}' | y_source == '{source}'")[[
                col for col in self.df.columns if col.startswith("y_")
            ]]
        for col in y_df.columns:
            y_df = y_df.rename(columns={col: col[2:]})
        # merge x and y nodes and keep only unique nodes
        out = pd.concat([
            x_df, y_df
        ], axis=0).query(f'source == "{source}"').drop_duplicates().reset_index(
            drop=True)

        return out


def build_KG(indices, relation, display_relation, x_id, x_type, x_name,
             x_source, y_id, y_type, y_name, y_source):
    df = pd.DataFrame('', columns=kg_columns, index=indices)

    df.relation = relation
    df.display_relation = display_relation

    df.x_id = x_id
    df.x_type = x_type
    df.x_name = x_name
    df.x_source = x_source

    df.y_id = y_id
    df.y_type = y_type
    df.y_name = y_name
    df.y_source = y_source

    kg = KnowledgeGraph(df)

    return kg
