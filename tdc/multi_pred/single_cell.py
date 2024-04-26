from collections.abc import Iterable
import pandas as pd
import tiledbsoma

from .anndata_dataset import DataLoader
from ..dataset_configs.cellxgene_config import CellXGeneConfig
from ..metadata import dataset_names
from ..resource.cellxgene_census import CensusResource


class CellXGeneTemplate(DataLoader):

    def __init__(self, name, path, print_stats, dsn=None):
        dsn = dataset_names["CellXGene"] if dsn is None else dsn
        super().__init__(name, path, print_stats, dsn)


class CellXGene(
        DataLoader
):  #TODO: create separate class for wrapping around a resource. as this is the only current use case it is left as is.

    def __init__(self, name, path="./data", print_stats=False):
        self.available_datasets = None
        self.resource = CensusResource(
        )  # only humans will be supported at this time
        self.is_collection = None
        self.genes = None
        self.config = CellXGeneConfig()
        self.data_processor = self.config.processing_callback
        if type(name) != list:
            if name not in self.get_available_datasets():
                raise ValueError("Invalid dataset name", name)
            self.is_collection = False
        elif not all(n in self.get_available_datasets() for n in name):
            raise Exception(
                "one or more input datasets in your list isn't in cellxgene available datasets",
                name)
        else:
            self.is_collection = True
        self.name = name
        assert self.is_collection is not None

    def get_data_for_dataset(self,
                             measurement_name=None,
                             value_filter=None,
                             value_adjustment=None):
        """Generator Expression"""
        measurement_name = measurement_name or "RNA"
        value_adjustment = value_adjustment or "raw"
        if self.genes is None:
            feature_presence_vector = self.resource.get_feature_dataset_presence_matrix_entry(
                self.name, measurement_name=measurement_name)
            genes_in_dataset = feature_presence_vector.values[0].tolist()
            self.genes = genes_in_dataset
        gene_filter = [idx for idx, v in enumerate(self.genes) if v]
        for slice in self.resource.query_measurement_matrix(
                value_filter=value_filter,
                measurement_name=measurement_name,
                value_adjustment=value_adjustment,
                gene_filter=gene_filter):
            filtered_df = self.data_processor(slice)
            yield filtered_df

    def get_data_for_collection(self,
                                measurement_name=None,
                                value_filter=None,
                                value_adjustment=None):
        """Generator Expression"""
        assert type(self.name) == list
        all_datasets = self.name
        self.genes = self.get_union_of_genes(measurement_name=measurement_name,
                                             all_datasets=all_datasets)
        yield from self.get_data_for_dataset(measurement_name=measurement_name,
                                             value_filter=value_filter,
                                             value_adjustment=value_adjustment)

    def get_union_of_genes(self, measurement_name=None, all_datasets=None):
        assert all_datasets is not None
        assert self.is_collection
        assert isinstance(all_datasets, Iterable)
        measurement_name = measurement_name or "RNA"
        gene_presence = None
        for ds in all_datasets:
            feature_presence_vector = self.resource.get_feature_dataset_presence_matrix_entry(
                ds, measurement_name=measurement_name)
            genes_in_dataset = feature_presence_vector.values[0].tolist()
            gene_presence = genes_in_dataset if gene_presence is None else [
                a | b for a, b in zip(genes_in_dataset, gene_presence)
            ]
        return gene_presence

    def get_data(self,
                 measurement_name=None,
                 value_filter=None,
                 value_adjustment=None):
        """TDC get_data() API implementation for the CELLXGENE data class. Can be a generator expression or return a traditional
        Pandas dataframe of union of all CELLXGENE entries of the queried dataset satisfying the provided conditions.

        Args:
            measurement_name (str, optional): measurement name to query, i.e. "RNA". Defaults to None.
            value_filter (str, optional): a valuer filter (obs) to apply to the query. Defaults to None.
            value_adjustment (str, optional): the type of count to obtain from count matricx for this measurement. Defaults to None.
            as_dataframe (bool, optional): Whether to return a fully concatendated dataframe or yield from the dataset. Defaults to False.

        Returns:
            pd.DataFrame: a dataframe if as_dataframe is True

        Yields:
            pd.DataFrame: yields chunks of the output datframe if as_dataframe is False

        WARNNG: requesting as_dataframe = True for too large a result may result in OOM or other process failure.
        """
        generator = self.get_data_for_collection(measurement_name=measurement_name, value_filter=value_filter, value_adjustment=value_adjustment) \
            if self.is_collection \
            else self.get_data_for_dataset(measurement_name=measurement_name, value_filter=value_filter, value_adjustment=value_adjustment)
        yield from generator

    def get_dataframe(self,
                      measurement_name=None,
                      value_filter=None,
                      value_adjustment=None,
                      debug=False):
        return pd.concat(self.get_data(measurement_name=measurement_name,
                                       value_filter=value_filter,
                                       value_adjustment=value_adjustment,
                                       debug=debug),
                         axis=0)

    def get_split(self,
                  measurement_name=None,
                  value_filter=None,
                  value_adjustment=None,
                  debug=False,
                  **kwargs):
        """TDC get_split() API implementation for the CELLXGENE data class. It assumes the dataset can fit in memory.
        Invokes the parent class get_split() function and supports configuring parameters for said class.

        WARNING: calling get_split() on too large a dataset can result in OOM or other process failure.
        """
        self.df = self.get_dataframe(measurement_name=measurement_name,
                                     value_filter=value_filter,
                                     value_adjustment=value_adjustment,
                                     debug=debug)
        assert len(self.df) > 1
        kwargs["data_ready"] = True
        return super().get_split(**kwargs)

    def get_available_datasets(self):
        if self.available_datasets is not None:
            return self.available_datasets
        self.available_datasets = self.resource.get_dataset_metadata()
        self.available_datasets = self.available_datasets[
            "dataset_title"].unique()
        return self.available_datasets
