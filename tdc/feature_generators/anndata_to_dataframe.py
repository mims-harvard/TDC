"""
Class for customizations when transforming anndata format objects to pandas.DataFrame or other formats
"""

import pandas as pd
import numpy as np
from .data_feature_generator import DataFeatureGenerator


class AnnDataToDataFrame(DataFeatureGenerator):

    @classmethod
    def anndata_to_df(cls, dataset=None, obs_cols=None):
        if dataset is None:
            raise ValueError("dataset must be specified")
        adata = dataset
        if not isinstance(adata.X, np.ndarray):
            adata.X = adata.X.todense()
        df_main = pd.DataFrame(adata.X if adata.X is not None else adata.X,
                               columns=adata.var_names,
                               index=adata.obs_names)
        dfobs = pd.DataFrame(adata.obs,
                             columns=adata.obs.keys(),
                             index=adata.obs.index)
        if obs_cols is None:
            return df_main
        elif obs_cols == "ALL":
            return df_main.merge(dfobs,
                                 left_index=True,
                                 right_index=True,
                                 how='left')
        elif isinstance(obs_cols, list):
            return df_main.merge(dfobs[obs_cols],
                                 left_index=True,
                                 right_index=True,
                                 how='left')
        else:
            raise ValueError("obs_cols must be a list of column names or 'ALL'")
