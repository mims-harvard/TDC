"""
Dataloader class for resource datasets
"""

import pandas as pd

from .. import base_dataset
from .. import metadata
from ..utils import load
from ..dataset_configs import config, config_map


class DataLoader(base_dataset.DataLoader, dict):
    """
    Dataloader class for resource datasets
    """

    def __init__(self, name, path="./data", dataset_names=None):
        dataset_names = dataset_names if dataset_names is not None else metadata.dataset_names[
            "Resource"]
        names = None
        if type(name) == list:
            names = name
        elif type(name) == str:
            assert name in metadata.resources, "{} is not a resource {}".format(
                name, metadata.resources.keys())
            names = [n for n in metadata.resources[name]["all"]]
        else:
            raise ValueError("expected a list of names or a single name. Got",
                             name)
        assert type(names) == list, "list expected"
        for n in names:
            self[n] = load.resource_dataset_load(n, path, dataset_names)
        for nsplit in metadata.resources[name].get("splits", []):
            self[nsplit] = {"splits": self[nsplit]}
        # raise Exception("keys", self.keys())
        self.config = config_map.ConfigMap().get(name)
        self.config = self.config()
        assert self.config is not None, "resource.DataLoader requires a corresponding config"
        assert isinstance(
            self.config, config.ResourceConfig
        ), "resource.DataLoader requires a ResourceConfig, got {}".format(
            type(self.config))
        # run transformations
        self.config.loader_setup_callback(self)
        # run data transformations
        if self.config.processing_callback is not None:
            self[self.config.df_key] = self.config.processing_callback(
                self[self.config.df_key])
        else:
            self[self.config.df_key] = self[self.config.df_key]

    def get_data(self, df_key=None, **kwargs):
        # TODO: can call parent's get_data(**kwargs) function if dataset not pre-loaded
        df_key = df_key or self.config.df_key
        assert df_key in self, "{} key hasn't been set in the loader, please set it by using the resource.DataLoader".format(
            df_key)
        return self[df_key]

    def get_split(self, split_key=None, **kwargs):
        # TODO: can call parent's get_split(**kwargs) function if splits not pre-loaded
        split_key = split_key or self.config.split_key
        assert split_key in self, "{} key hasn't been set in the loader, please set it by using the resource.DataLoader".format(
            split_key)
        return self[split_key]
