# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

from .base_group import BenchmarkGroup

class drugcombo_group(BenchmarkGroup):
	"""create a drug combination benchmark group
		
	Args:
	    path (str, optional): path to save/load benchmarks
	"""

	def __init__(self, path = './data'):
		"""create a drug combination benchmark group
		"""
		super().__init__(name = 'DrugCombo_Group', path = path, file_format = 'pkl')