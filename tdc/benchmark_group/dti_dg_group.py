"""DTI Domain Generalization Benchmark Group
"""
from .base_group import BenchmarkGroup

class dti_dg_group(BenchmarkGroup):

	"""Create a DTI domain generalization benchmark group
	"""
	
	def __init__(self, path = './data'):
		"""Create a DTI domain generalization benchmark group
		
		Args:
		    path (str, optional): path to save/load benchmarks
		"""
		super().__init__(name = 'DTI_DG_Group', path = path, file_format = 'csv')