# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

from .base_group import BenchmarkGroup


class drugcombo_group(BenchmarkGroup):
    """create a drug combination benchmark group

    Args:
        path (str, optional): path to save/load benchmarks
    """

    def __init__(self, path="./data"):
        """create a drug combination benchmark group"""
        super().__init__(name="DrugCombo_Group", path=path, file_format="pkl")

    def get_cell_line_meta_data(self):
        import os
        from ..utils.load import download_wrapper
        from ..utils import load_dict
        name = download_wrapper('drug_comb_meta_data', self.path,
                                ['drug_comb_meta_data'])
        return load_dict(os.path.join(self.path, name + '.pkl'))
