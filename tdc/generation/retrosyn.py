"""Summary
"""
import warnings
warnings.filterwarnings("ignore")

from . import generation_dataset
from ..metadata import dataset_names

class RetroSyn(generation_dataset.PairedDataLoader):

	"""Summary
	"""
	
	def __init__(self, name, path = './data', print_stats = False, input_name = 'product', output_name = 'reactant'): 
		"""Summary
		
		Args:
		    name (TYPE): Description
		    path (str, optional): Description
		    print_stats (bool, optional): Description
		    input_name (str, optional): Description
		    output_name (str, optional): Description
		"""
		super().__init__(name, path, print_stats, input_name, output_name)