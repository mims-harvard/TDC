# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

from .base_group import BenchmarkGroup


class dti_dg_group(BenchmarkGroup):
    """Create a DTI domain generalization benchmark group

    Args:
        path (str, optional): path to save/load benchmarks
    """

    def __init__(self, path="./data"):
        """Create a DTI domain generalization benchmark group"""
        super().__init__(name="DTI_DG_Group", path=path, file_format="csv")
