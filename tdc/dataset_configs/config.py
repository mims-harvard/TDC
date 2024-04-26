"""General class for representing dataset configs"""

import pandas as pd

from ..base_dataset import DataLoader
from ..feature_generators import resource, base


class ConfigBase(dict):
    """Class for representing configs"""
    pass


class DatasetConfig(ConfigBase):
    """Class for representing dataset configs"""

    def __init__(self,
                 dataset_name=None,
                 data_processing_class=None,
                 functions_to_run=None,
                 args_for_functions=None,
                 var_map=None,
                 **kwargs):
        self.dataset_name = dataset_name
        self.data_processing_class = data_processing_class
        self.functions_to_run = functions_to_run
        self.args_for_functions = args_for_functions
        self.var_map = var_map
        self["kwargs"] = kwargs

    @property
    def name(self):
        return self.dataset_name

    @property
    def processing_class(self):
        return self.data_processing_class

    @property
    def functions(self):
        return self.functions_to_run

    @property
    def args(self):
        return self.args_for_functions

    @property
    def functions_and_args(self):
        return zip(self.functions, self.args)

    @property
    def processing_callback(self):
        """Returns a callback for processing a dataset according to this config"""
        return (lambda dataset: self.data_processing_class.process_data(
            dataset, self.functions, self.args)
               ) if self.data_processing_class is not None else None

    @property
    def tdc_cols_callback(self, dataset_type=None):
        """Add the standard  TDC columns (e.g., 'X1', 'X2', ...) to a given DataFrame."""

        def bi_pred_var_map_helper(dataset):
            assert dataset is not None, "Must provide a dataset."
            dataset["X1"] = dataset[self.var_map["X1"]]
            dataset["X2"] = dataset[self.var_map["X2"]]
            dataset["ID1"] = dataset[self.var_map["ID1"]]
            dataset["ID2"] = dataset[self.var_map["ID2"]]
            return dataset

        dataset_type = dataset_type if dataset_type is not None else "bi_pred_dataset"
        if dataset_type == "bi_pred_dataset" and self.var_map is not None:
            return bi_pred_var_map_helper
        else:
            return None


class LoaderConfig(DatasetConfig):
    """Class for representing configs for DataLoader instances
    
    Requires DataLoader parent class to inherit from dictionary
    LoaderConfig allows DataLoader instance variables to be used
    as parameters in feature generation"""

    def __init__(self,
                 keys=None,
                 loader_functions=None,
                 loader_args=None,
                 feature_class=None,
                 dataset_name=None,
                 data_processing_class=None,
                 functions_to_run=None,
                 args_for_functions=None,
                 var_map=None,
                 **kwargs):
        if feature_class is not None:
            assert isinstance(feature_class, base.FeatureGenerator)
            self.feature_class = feature_class
        super().__init__(dataset_name=None,
                         data_processing_class=None,
                         functions_to_run=None,
                         args_for_functions=None,
                         var_map=None,
                         **kwargs)
        self.loader_functions = loader_functions if loader_functions else []
        self.loader_args = loader_args if loader_args else []
        self.keys = keys

    @property
    def loader_setup_callback(self):
        """Returns a callback for processing a dataloader according to this config"""
        return lambda loader: self.feature_class.process_loader(
            loader, self.keys, self.loader_functions, self.loader_args
        ) if self.feature_class is not None else None


class ResourceConfig(LoaderConfig):
    """Class for representing configs for Resource DataLoader instances"""

    def __init__(self,
                 feature_class,
                 dataset_name=None,
                 data_processing_class=None,
                 functions_to_run=None,
                 args_for_functions=None,
                 var_map=None,
                 **kwargs):
        assert isinstance(feature_class, resource.ResourceFeatureGenerator)
        super().__init__(feature_class=feature_class, **kwargs)

    @property
    def df_key(self):
        return "df"

    @property
    def split_key(self):
        return "split"
