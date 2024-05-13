from .brown_mdm2_ace2_12ca5_config import BrownProteinPeptideConfig
from .opentargets_dti import OpentargetsDTI
from .scperturb_config import SCPerturb, SCPerturb_Gene

scperturb_datasets = [
    "scperturb_drug_AissaBenevolenskaya2021",
    "scperturb_drug_SrivatsanTrapnell2020_sciplex2",
    "scperturb_drug_SrivatsanTrapnell2020_sciplex3",
    "scperturb_drug_SrivatsanTrapnell2020_sciplex4",
    "scperturb_drug_ZhaoSims2021",
]

scperturb_gene_datasets = [
    "scperturb_gene_NormanWeissman2019",
    "scperturb_gene_ReplogleWeissman2022_rpe1",
    "scperturb_gene_ReplogleWeissman2022_k562_essential",
]


class ConfigMap(dict):
    """
    The ConfigMap stores key-value pairs where the key is a dataset string name and the value is a config class.
    """

    def __init__(self):
        self["brown_mdm2_ace2_12ca5"] = BrownProteinPeptideConfig
        for ds in scperturb_datasets:
            self[ds] = SCPerturb
        for ds in scperturb_gene_datasets:
            self[ds] = SCPerturb_Gene
        self["opentargets_dti"] = OpentargetsDTI
