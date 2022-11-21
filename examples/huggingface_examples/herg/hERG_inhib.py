#!/usr/bin/env python
# coding: utf-8

get_ipython().system(' pip show PyTDC')


RANDOM_SEED = 42


import pandas as pd

from tdc import utils as tdc_utils
from tdc.single_pred import Tox

from DeepPurpose import utils as dp_utils, CompoundPred


X, y = Tox(name = 'herg_central', label_name="hERG_inhib").get_data(format = 'DeepPurpose')


X, y = X[:100], y[:100]


pd.Series(y).value_counts()


hparams = {
    "drug_encoding": "MPNN",
    "lr": 0.001,
    "batch_size": 128,
    "mpnn_hidden_size": 32,
    "mpnn_depth": 2,
    "train_epochs": 3,
}


get_ipython().run_cell_magic('time', '', '\ntrain, val, test = dp_utils.data_process(X_drug=X, y = y, drug_encoding=hparams["drug_encoding"], \n                                      random_seed=RANDOM_SEED)\n')


config = dp_utils.generate_config(
    drug_encoding=hparams["drug_encoding"], 
    train_epoch=hparams['train_epochs'], 
    LR=hparams['lr'], 
    batch_size=hparams['batch_size'],
    mpnn_hidden_size=hparams['mpnn_hidden_size'],
    mpnn_depth=hparams['mpnn_depth']
)


model = CompoundPred.model_initialize(**config)


model.train(train, val, test)


model.save_model('./tutorial_model')




