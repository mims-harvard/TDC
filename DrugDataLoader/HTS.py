import pandas as pd
import numpy as np
import os, sys, json 
import warnings
warnings.filterwarnings("ignore")

from DrugDataLoader import utils, DrugProperty

class DataLoader(DrugProperty.DataLoader):
	def __init__(self, name, path = './data', target = None, print_stats = True):
		super().__init__(name, path, target)