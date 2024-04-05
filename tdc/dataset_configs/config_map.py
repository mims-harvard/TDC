from .pentelute_mdm2_ace2_12ca5_config import PenteluteProteinPeptideConfig
from .scperturb_config import SCPerturb

scperturb_datasets = [
    "scperturb_drug_AissaBenevolenskaya2021",
    "scperturb_drug_SrivatsanTrapnell2020_sciplex2",
    "scperturb_drug_SrivatsanTrapnell2020_sciplex3",
    "scperturb_drug_SrivatsanTrapnell2020_sciplex4",
    "scperturb_drug_ZhaoSims2021",
]


class ConfigMap(dict):
    """
    The ConfigMap stores key-value pairs where the key is a dataset string name and the value is a config class.
    """

    def __init__(self):
        self["pentelute_mdm2_ace2_12ca5"] = PenteluteProteinPeptideConfig
        for ds in scperturb_datasets:
            self[ds] = SCPerturb
