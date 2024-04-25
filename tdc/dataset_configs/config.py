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
        return lambda dataset: self.data_processing_class.process_data(
            dataset, self.functions, self.args
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
    
    def __init__(self, loader_functions=None, loader_args=None, feature_class=None, **kwargs):
        if feature_class is not None:
            assert isinstance(feature_class, base.FeatureGenerator)
            self.feature_class = feature_class
        super().__init__(**kwargs)
        self.data = None
        self.loader_functions = loader_functions if loader_functions else []
        self.loader_args = loader_args if loader_args else []
        self.dataloader = None
    
    @property
    def df(self):
        return self.data if self.data is not None else self.get_data()
    
    def get_data(self):
        if not self.dataloader:
            raise Exception("DataLoader not set. Call set_loader()")
        elif not self.data_config:
            self.set_loader(self.dataloader)
        init_frame = None
        try:
            init_frame = self.dataloader.df
        except:
            init_frame = pd.DataFrame()
        self.data = self.data_config.processing_callback(init_frame)
        return self.data

    def set_loader(self, dataloader):
        assert isinstance(dataloader, DataLoader)
        assert isinstance(dataloader, dict)
        self.dataloader = dataloader
        for k,v in self.loader_args.items():
            if isinstance(v, tuple) and v[0]=="self" and len(v)==2:
                self.loader_args[k] = self.dataloader[v[1]]
        self.data_config = DatasetConfig(data_processing_class=self.feature_class, functions_to_run=self.loader_functions, args_for_functions=self.loader_args)
        
class ResourceConfig(LoaderConfig):
    """Class for representing configs for Resource DataLoader instances"""
    
    def __init__(self, dataloader, feature_class, **kwargs):
        assert isinstance(feature_class, resource.ResourceFeatureGenerator) 
        super().__init__(dataloader, feature_class=feature_class, **kwargs)