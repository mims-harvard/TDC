"""Drug combination benchmark group
"""
from .base_group import BenchmarkGroup

class drugcombo_group(BenchmarkGroup):

	"""create a drug combination benchmark group
	"""
	
	def __init__(self, path = './data'):
		"""create a drug combination benchmark group
		
		Args:
		    path (str, optional): path to save/load benchmarks
		"""
		super().__init__(name = 'DrugCombo_Group', path = path, file_format = 'pkl')