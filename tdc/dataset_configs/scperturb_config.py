from .config import DatasetConfig
from ..feature_generators.anndata_to_dataframe import AnnDataToDataFrame


class SCPerturb(DatasetConfig):
    """Configuration for the scPerturb datasets"""

    def __init__(self):
        super(SCPerturb, self).__init__(
            data_processing_class=AnnDataToDataFrame,
            functions_to_run=["anndata_to_df"],
            args_for_functions=[
                {
                    "obs_cols": [
                        "ncounts", 'celltype', 'cell_line', 'cancer', 'disease',
                        'tissue_type', 'perturbation', 'perturbation_type',
                        'ngenes'
                    ],
                },
            ],
        )


class SCPerturb_Gene(DatasetConfig):
    """Configuration for the scPerturb genetic perturbation datasets"""

    def __init__(self):
        super(SCPerturb_Gene, self).__init__(
            data_processing_class=AnnDataToDataFrame,
            functions_to_run=["anndata_to_df"],
            args_for_functions=[
                {
                    "obs_cols": [
                        'UMI_count', 'cancer', 'cell_line', 'disease',
                        'guide_id', 'ncounts', 'ngenes', 'nperts', 'organism',
                        'percent_mito', 'percent_ribo', 'perturbation',
                        'perturbation_type', 'tissue_type'
                    ],
                },
            ],
        )
