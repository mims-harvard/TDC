# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

from .base_group import BenchmarkGroup

class dti_dg_group(BenchmarkGroup):
	def __init__(self, path = './data'):
		super().__init__(name = 'DTI_DG_Group', path = path, file_format = 'csv')