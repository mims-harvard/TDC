# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

from .base_group import BenchmarkGroup


class admet_group(BenchmarkGroup):
    """Create ADMET Group Class object.

    Args:
            path (str, optional): the path to store/retrieve the ADMET group datasets.
    """

    def __init__(self, path="./data"):
        """Create an ADMET benchmark group class."""
        super().__init__(name="ADMET_Group", path=path, file_format="csv")
