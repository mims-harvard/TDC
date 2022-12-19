#!/usr/bin/env python
# coding: utf-8

get_ipython().system(" pip show PyTDC")


RANDOM_SEED = 42


import pandas as pd

from tdc import utils as tdc_utils
from tdc.single_pred import Tox

from DeepPurpose import utils as dp_utils, CompoundPred


# ## Prepare Data

X, y = Tox(name="herg_central", label_name="hERG_inhib").get_data(format="DeepPurpose")


X, y = X[:100], y[:100]


pd.Series(y).value_counts()


# ## Train

import io
from contextlib import redirect_stdout
from functools import lru_cache

import re

import pandas as pd
from typing import List


def parse_train_log(log_lines: List[str]):
    """
    Parses the log lines from `model.train` function call
    """
    train_stats, val_stats = [], []

    for i, l in enumerate(log_lines):
        m = re.match(
            r"Training at Epoch (?P<epoch>\d+) iteration (?P<iter>\d+) with loss (?P<loss>\d+\.\d+). Total time 0.0 hours",
            l,
        )
        if m:
            train_stats.append(
                {
                    "epoch": int(m.group("epoch")),
                    "train_iter": int(m.group("iter")),
                    "train_loss": float(m.group("loss")),
                }
            )

        m = re.match(
            r"Validation at Epoch (?P<epoch>\d+) , AUROC: (?P<aucroc>\d+\.\d+) , AUPRC: (?P<aupr>\d+\.\d+) , F1: (?P<f1>\d+\.\d+)",
            l,
        )
        if m:
            val_stats.append(
                {
                    "epoch": int(m.group("epoch")),
                    "val_aucroc": float(m.group("aucroc")),
                    "val_aupr": float(m.group("aupr")),
                    "val_f1": float(m.group("f1")),
                }
            )

        m = re.match(
            r"Testing AUROC: (?P<aucroc>\d+\.\d+) , AUPRC: (?P<aupr>\d+\.\d+) , F1: (?P<f1>\d+\.\d+)",
            l,
        )
        if m:
            test_stats = {
                "test_aucroc": float(m.group("aucroc")),
                "test_aupr": float(m.group("aupr")),
                "test_f1": float(m.group("f1")),
            }

    train_stats_df = pd.DataFrame.from_records(train_stats)
    val_stats_df = pd.DataFrame.from_records(val_stats)

    train_val_stats_df = pd.merge(
        left=train_stats_df, right=val_stats_df, on="epoch", validate="1:1"
    )

    return {"train_val": train_val_stats_df, "test": test_stats}


# @lru_cache() # doesn't work with Ray Tune
def prepare_data(drug_encoding):
    return dp_utils.data_process(
        X_drug=X, y=y, drug_encoding=drug_encoding, random_seed=RANDOM_SEED
    )


def train_fn(hparams):

    train, val, test = prepare_data(hparams["drug_encoding"])

    config = dp_utils.generate_config(
        drug_encoding=hparams["drug_encoding"],
        train_epoch=int(hparams["train_epochs"]),
        LR=hparams["lr"],
        batch_size=int(hparams["batch_size"]),
        mpnn_hidden_size=int(hparams["mpnn_hidden_size"]),
        mpnn_depth=int(hparams["mpnn_depth"]),
    )

    model = CompoundPred.model_initialize(**config)

    f = io.StringIO()
    with redirect_stdout(f):
        model.train(train, val, test)
    out = f.getvalue()

    results = parse_train_log(out.split("\n"))

    model.save_model("./tutorial_model.pt")
    results["train_val"].to_csv("train_val_stats.csv", index=False)
    return {**results["train_val"].to_dict(orient="records")[-1], **results["test"]}


# ## Tune

from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch

from ray import air
from ray.air import session
from ray.air.callbacks.mlflow import MLflowLoggerCallback


search_space = {
    "drug_encoding": "MPNN",
    "lr": 0.001,
    "batch_size": 128,
    "mpnn_hidden_size": tune.uniform(8, 32),
    "mpnn_depth": 2,
    "train_epochs": 3,
}

bayesopt = BayesOptSearch(metric="train_loss", mode="min")
tuner = tune.Tuner(
    train_fn,
    tune_config=tune.TuneConfig(search_alg=bayesopt, num_samples=3),
    run_config=air.RunConfig(
        name="mlflow",
        callbacks=[
            MLflowLoggerCallback(
                tracking_uri="./mlruns",
                experiment_name="test",
                save_artifact=True,
            )
        ],
    ),
    param_space=search_space,
)

analysis = tuner.fit()


# Explore the tuning results on mlflow dashboard as well. It can be started by running `mlflow ui --backend-store-uri examples/huggingface_examples/herg/mlruns/` in terminal. Any files saved to local disk during training can be found in the corresponding run in the `examples/huggingface_examples/herg/mlruns/` directory.

# ## Export to Huggingface Hub


# ## Appendix

# train_fn({
#     "drug_encoding": "MPNN",
#     "lr": 0.001,
#     "batch_size": 128,
#     "mpnn_hidden_size": 32,
#     "mpnn_depth": 2,
#     "train_epochs": 3,
# })
