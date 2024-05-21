from .multi_pred_dataset import DataLoader as DL
from ..dataset_configs.config_map import ConfigMap
from ..feature_generators.anndata_to_dataframe import AnnDataToDataFrame


class DataLoader(DL):

    def __init__(self, name, path, print_stats, dataset_names):
        super(DataLoader, self).__init__(name, path, print_stats, dataset_names)
        self.adata = self.df  # this is in AnnData format
        cmap = ConfigMap()
        self.cmap = cmap
        self.config = cmap.get(name)
        if self.config is None:
            # default to converting adata to dataframe as is
            self.df = AnnDataToDataFrame.anndata_to_df(self.adata)
        else:
            cf = self.config()
            self.df = cf.processing_callback(self.adata)
