from collections.abc import Iterable

from .anndata_dataset import DataLoader
from .cellxgene_metadata.collection_to_datasets import collection_to_datasets
from ..metadata import dataset_names
from ..resource.cellxgene_census import CensusResource


class CellXGeneTemplate(DataLoader):

    def __init__(self, name, path, print_stats, dsn=None):
        dsn = dataset_names["CellXGene"] if dsn is None else dsn
        super().__init__(name, path, print_stats, dsn)


class SingleCellPrediction(CellXGeneTemplate):

    def __init__(self, name, path="./data", print_stats=False):
        super().__init__(name, path, print_stats)


class CellXGene(DataLoader):  #TODO: create separate class for wrapping around a resource. as this is the only current use case it is left as is.

    def __init__(self, name, path="./data", print_stats=False):
        self.available_datasets = None
        self.resource = CensusResource() # only humans will be supported at this time
        self.is_collection = None
        self.genes = None
        if name not in collection_to_datasets:
            if name not in self.get_available_datasets():
                raise ValueError("Invalid dataset name", name)
            self.is_collection = False
        else:
            self.is_collection = True
        self.name = name
        assert self.is_collection is not None
        
    def get_data_for_dataset(self, measurement_name = None, value_filter = None, value_adjustment = None):
        """Returns a generator"""
        if self.genes is None:
            feature_presence_vector = self.resource.get_feature_dataset_presence_matrix_entry(self.name, measurement_name=measurement_name)
            genes_in_dataset = feature_presence_vector.columns[feature_presence_vector.loc[feature_presence_vector]]
            self.genes = genes_in_dataset.tolist()
        if value_filter is not None:
            for slice in self.resource.query_measurement_matrix(
                value_filter=value_filter,
                measurement_name=measurement_name,
                value_adjustment=value_adjustment
            ):
                df = slice[self.genes]
                filtered_df = df[df.any(axis=1)]
                yield filtered_df
        else:
            # TODO: yield from the measurement matrix
            pass
        
    def get_data_for_collection(self, measurement_name = None, value_filter = None, value_adjustment = None):
        """Returns a generator"""
        assert self.name in collection_to_datasets
        all_datasets = collection_to_datasets[self.name]
        self.genes = self.get_union_of_genes(measurement_name = measurement_name, all_datasets = all_datasets)
        yield from self.get_data_for_dataset(measurement_name=measurement_name, value_filter=value_filter, value_adjustment=value_adjustment)
            
    def get_union_of_genes(self, measurement_name = None, all_datasets = None):
        assert all_datasets is not None
        assert self.is_collection
        assert isinstance(all_datasets, Iterable)
        sset = set()
        for ds in all_datasets:
            feature_presence_vector = self.resource.get_feature_dataset_presence_matrix_entry(ds, measurement_name=measurement_name)
            genes_in_dataset = feature_presence_vector.columns[feature_presence_vector.loc[feature_presence_vector]]
            sset.union(set(genes_in_dataset.tolist()))
        return list(sset) 
            
    def get_available_datasets(self):
        if self.available_datasets is not None:
            return self.available_datasets
        self.available_datasets = self.resource.get_dataset_metadata()
        self.available_datasets = set(self.available_datasets.index)
        return self.available_datasets