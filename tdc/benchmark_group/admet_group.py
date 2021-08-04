from .base_group import BenchmarkGroup

class admet_group(BenchmarkGroup):
	def __init__(self, path = './data'):
		super().__init__(name = 'ADMET_Group', path = path, file_format = 'csv')