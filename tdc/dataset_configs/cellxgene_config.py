from .config import DatasetConfig
from ..feature_generators.cellxgene_generator import CellXGeneFeatureGenerator


class CellXGeneConfig(DatasetConfig):

    def __init__(self):
        super(CellXGeneConfig, self).__init__(
            data_processing_class=CellXGeneFeatureGenerator,
            functions_to_run=[
                "get_dense_soma_dataframe", "format_cellxgene_dataframe"
            ],
            args_for_functions=[{}, {}, {}],
        )
