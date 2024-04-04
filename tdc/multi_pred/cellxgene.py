from .anndata_dataset import DataLoader
from ..metadata import dataset_names

class CellXGene(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs["dataset_names"] = dataset_names["CellXGene"]
        super().__init__(*args, **kwargs)