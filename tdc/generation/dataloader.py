import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from ..utils import *
from . import generation_dataset
from ..metadata import dataset_names

class Reaction(generation_dataset.PairedDataLoader):

	def __init__(self, name, path = './data', print_stats = False, input_name = 'reactant', output_name = 'product'): 
		super().__init__(name, path, print_stats, input_name, output_name)

class RetroSyn(generation_dataset.PairedDataLoader):

	def __init__(self, name, path = './data', print_stats = False, input_name = 'product', output_name = 'reactant'): 
		super().__init__(name, path, print_stats, input_name, output_name)


class MolGen(generation_dataset.DataLoader):

	def __init__(self, name, path = './data', print_stats = False, column_name = 'smiles'): 
		super().__init__(name, path, print_stats, column_name)

