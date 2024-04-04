from .config import DatasetConfig
from ..feature_generators.anndata_to_dataframe import AnnDataToDataFrame


class PenteluteProteinPeptideConfig(DatasetConfig):
    """Configuration for the pentelute-protein-peptide datasets"""

    def __init__(self):
        super(PenteluteProteinPeptideConfig, self).__init__(
            dataset_name="pentelute_mdm2_ace2_12ca5",
            data_processing_class=AnnDataToDataFrame,
            functions_to_run=[
                "anndata_to_df"
            ],
            args_for_functions=[{
                "obs_cols": ["ncounts", 'celltype', 'cell_line', 'cancer', 'disease', 'tissue_type', 'perturbation', 'perturbation_type', 'ngenes'],
            },
            ],
        )
