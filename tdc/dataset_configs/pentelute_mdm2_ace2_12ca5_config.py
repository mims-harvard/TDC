from .config import DatasetConfig
from ..utils.protein_data_utils import ProteinDataUtils

class PenteluteProteinPeptideConfig(DatasetConfig):
    """Configuration for the pentelute-protein-peptide datasets"""
    def __init__(self):
        super(PenteluteProteinPeptideConfig, self).__init__(
            dataset_name="pentelute_mdm2_ace2_12ca5",
            data_processing_class = ProteinDataUtils,
            functions_to_run=["autofill_identifier", "create_range", "insert_protein_sequence"],
            args_for_functions=[
                {
                    "autofill_column": "Name",
                    "key_column": "Sequence",
                },
                {
                    "column": "KD (nM)",
                    "keys": ["Putative binder"],
                    "subs": [0]
                },
                {
                    "gene_column": "Protein Target"
                }
            ],
            var_map = {
                "X1": "Sequence",
                "X2": "protein_or_rna_sequence",
                "ID1": "Name",
                "ID2": "Protein Target",
            },
        )