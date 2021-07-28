"""Summary
"""
import warnings
warnings.filterwarnings("ignore")

from . import generation_dataset
from ..metadata import dataset_names

class MolGen(generation_dataset.DataLoader):

	"""Summary
	"""
	
	def __init__(self, name, path = './data', print_stats = False, column_name = 'smiles'): 
		"""Summary
		
		Args:
		    name (TYPE): Description
		    path (str, optional): Description
		    print_stats (bool, optional): Description
		    column_name (str, optional): Description
		"""
		super().__init__(name, path, print_stats, column_name)

