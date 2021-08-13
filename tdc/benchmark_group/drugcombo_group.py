# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

from .base_group import BenchmarkGroup

class drugcombo_group(BenchmarkGroup):
	def __init__(self, path = './data'):
		super().__init__(name = 'DrugCombo_Group', path = path, file_format = 'pkl')