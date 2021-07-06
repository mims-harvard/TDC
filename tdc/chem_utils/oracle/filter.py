import pickle 
import numpy as np 
import re
import os.path as op
import math
from collections import defaultdict, Iterable
from abc import abstractmethod
from functools import partial
from typing import List
import time
import os 

try:
    from sklearn import svm
    # from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, precision_score, recall_score, accuracy_score
except:
    ImportError("Please install sklearn by 'conda install -c anaconda scikit-learn' or 'pip install scikit-learn '! ")

try: 
  import rdkit
  from rdkit import Chem, DataStructs
  from rdkit.Chem import AllChem
  from rdkit.Chem import Descriptors
  import rdkit.Chem.QED as QED
  from rdkit import rdBase
  rdBase.DisableLog('rdApp.error')
  from rdkit.Chem import rdMolDescriptors
  from rdkit.six.moves import cPickle
  from rdkit.six import iteritems
  from rdkit.Chem.Fingerprints import FingerprintMols
  from rdkit.Chem import MACCSkeys
except:
  raise ImportError("Please install rdkit by 'conda install -c conda-forge rdkit'! ")   

try:
    from scipy.stats.mstats import gmean
except:
    raise ImportError("Please install rdkit by 'pip install scipy'! ") 


try:
    import networkx as nx 
except:
    raise ImportError("Please install networkx by 'pip install networkx'! ")    

# from ..utils import oracle_load, print_sys, install
from tdc.utils import oracle_load, print_sys, install





class MolFilter:
  """ Molecule Filter: filter Molecule based on user-specified condition 

  Args:
    filters: 
    property_filters_flag: bool, 
    HBA: [lower_bound, upper_bound]
    HBD: [lower_bound, upper_bound]
    LogP: [lower_bound, upper_bound]
    MW: [lower_bound, upper_bound],   Molecule weight
    Rot: [lower_bound, upper_bound]
    TPSA: [lower_bound, upper_bound]

  Returns:
    list of SMILES strings that pass the filter. 

  """  
  # MIT License: Checkout https://github.com/PatWalters/rd_filters
  def __init__(self, filters = 'all', property_filters_flag = True, HBA = [0, 10], HBD = [0, 5], LogP = [-5, 5], MW = [0, 500], Rot = [0, 10], TPSA = [0, 200]):
    try:
        from rd_filters.rd_filters import RDFilters, read_rules
    except:
        install('git+https://github.com/PatWalters/rd_filters.git')
        from rd_filters.rd_filters import RDFilters, read_rules
        
    import pkg_resources
    self.property_filters_flag = property_filters_flag
    all_filters = ['BMS', 'Dundee', 'Glaxo', 'Inpharmatica', 'LINT', 'MLSMR', 'PAINS', 'SureChEMBL']
    if filters == 'all':
      filters = all_filters
    else:
      if isinstance(filters, str):
        filters = [filters]
      if isinstance(filters, list):
        ## a set of filters
        for i in filters:
          if i not in all_filters:
            raise ValueError(i + " not found; Please choose from a list of available filters from 'BMS', 'Dundee', 'Glaxo', 'Inpharmatica', 'LINT', 'MLSMR', 'PAINS', 'SureChEMBL'")

    alert_file_name = pkg_resources.resource_filename('rd_filters', "data/alert_collection.csv")
    rules_file_path = pkg_resources.resource_filename('rd_filters', "data/rules.json")
    self.rf = RDFilters(alert_file_name)
    self.rule_dict = read_rules(rules_file_path)
    self.rule_dict['Rule_Inpharmatica'] = False
    for i in filters:
      self.rule_dict['Rule_'+ i] = True
    
    if self.property_filters_flag:
        self.rule_dict['HBA'], self.rule_dict['HBD'], self.rule_dict['LogP'], self.rule_dict['MW'], self.rule_dict['Rot'], self.rule_dict['TPSA'] = HBA, HBD, LogP, MW, Rot, TPSA
    else:
        del self.rule_dict['HBA'], self.rule_dict['HBD'], self.rule_dict['LogP'], self.rule_dict['MW'], self.rule_dict['Rot'], self.rule_dict['TPSA']
    print_sys("MolFilter is using the following filters:")

    for i,j in self.rule_dict.items():
      if i[:4] == 'Rule':
        if j:
          print_sys(i + ': ' + str(j))
      else:
        print_sys(i + ': ' + str(j))
    rule_list = [x.replace("Rule_", "") for x in self.rule_dict.keys() if x.startswith("Rule") and self.rule_dict[x]]
    rule_str = " and ".join(rule_list)
    self.rf.build_rule_list(rule_list)

  def __call__(self, input_data):
    import multiprocessing as mp
    from multiprocessing import Pool
    import pandas as pd

    if isinstance(input_data, str):
      input_data = [input_data]
    elif not isinstance(input_data, (list, np.ndarray, np.generic)):
      raise ValueError('Input must be a list/numpy array of SMILES or one SMILES string!')

    input_data = list(tuple(zip(input_data, list(range(len(input_data))))))

    num_cores = int(mp.cpu_count())
    p = Pool(num_cores)

    res = list(p.map(self.rf.evaluate, input_data))
    
    if self.property_filters_flag:
    
        df = pd.DataFrame(res, columns=["SMILES", "NAME", "FILTER", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"])
        df_ok = df[
            (df.FILTER == "OK") &
            df.MW.between(*self.rule_dict["MW"]) &
            df.LogP.between(*self.rule_dict["LogP"]) &
            df.HBD.between(*self.rule_dict["HBD"]) &
            df.HBA.between(*self.rule_dict["HBA"]) &
            df.TPSA.between(*self.rule_dict["TPSA"]) &
            df.Rot.between(*self.rule_dict["Rot"])
            ]
        
    else:
        df = pd.DataFrame(res, columns=["SMILES", "NAME", "FILTER", "MW", "LogP", "HBD", "HBA", "TPSA", "Rot"])
        df_ok = df[
            (df.FILTER == "OK")
            ]
    return df_ok.SMILES.values






