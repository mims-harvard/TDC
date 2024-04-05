from .anndata_dataset import DataLoader
from ..metadata import dataset_names


class CellXGene(DataLoader):

    def __init__(self, name, path, print_stats, dsn=None):
        dsn = dataset_names["CellXGene"] if dsn is None else dsn
        super().__init__(name, path, print_stats, dsn)


class SingleCellPrediction(CellXGene):

    def __init__(self, name, path="./data", print_stats=False):
        super().__init__(name, path, print_stats)
