# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from ..utils import bm_group_load, print_sys, fuzzy_search
from ..utils import (
    create_fold,
    create_fold_setting_cold,
    create_combination_split,
    create_fold_time,
    create_scaffold_split,
    create_group_split,
)
from ..metadata import (
    get_task2category,
    bm_metric_names,
    benchmark_names,
    bm_split_names,
    docking_target_info,
)
from ..evaluator import Evaluator


class BenchmarkGroup:
    """Boilerplate of benchmark group class. It downloads, processes, and loads a set of benchmark classes along with their splits. It also provides evaluators and train/valid splitters."""

    def __init__(self, name, path="./data", file_format="csv"):
        """create a benchmark group class object

        Args:
                name (str): the name of the benchmark group class
                path (str, optional): the path to save/load the benchkmark group dataset
                file_format (str, optional): designated file format for each dataset in the benchmark group

        """
        self.name = bm_group_load(name, path)
        self.path = os.path.join(path, self.name)
        self.datasets = benchmark_names[self.name]
        self.dataset_names = []
        self.file_format = file_format

        for task, datasets in self.datasets.items():
            for dataset in datasets:
                self.dataset_names.append(dataset)

    def __iter__(self):
        """iterator implementation to iterate over all benchmarks in the benchmark group

        Returns:
            BenchmarkGroup: self
        """
        self.index = 0
        self.num_datasets = len(self.dataset_names)
        return self

    def __next__(self):
        """iterator implementation to define the next benchmark

        Returns:
            dict: a dictionary of key values in a benchmark, namely the train_val file, test file and benchmark name

        Raises:
            StopIteration: stop when exceed the number of benchmarks
        """
        if self.index < self.num_datasets:
            dataset = self.dataset_names[self.index]
            print_sys("--- " + dataset + " ---")

            data_path = os.path.join(self.path, dataset)
            if not os.path.exists(data_path):
                os.mkdir(data_path)
            if self.file_format == "csv":
                train = pd.read_csv(os.path.join(data_path, "train_val.csv"))
                test = pd.read_csv(os.path.join(data_path, "test.csv"))
            elif self.file_format == "pkl":
                train = pd.read_pickle(os.path.join(data_path, "train_val.pkl"))
                test = pd.read_pickle(os.path.join(data_path, "test.pkl"))
            self.index += 1
            return {"train_val": train, "test": test, "name": dataset}
        else:
            raise StopIteration

    def get_train_valid_split(self, seed, benchmark, split_type="default"):
        """obtain training and validation split given a split type from train_val file

        Args:
            seed (int): the random seed of the data split
            benchmark (str): name of the benchmark
            split_type (str, optional): name of the split

        Returns:
            pd.DataFrame: the training and validation files

        Raises:
            NotImplementedError: split method not implemented
        """
        print_sys("generating training, validation splits...")
        dataset = fuzzy_search(benchmark, self.dataset_names)
        data_path = os.path.join(self.path, dataset)
        if self.file_format == "csv":
            train_val = pd.read_csv(os.path.join(data_path, "train_val.csv"))
        elif self.file_format == "pkl":
            train_val = pd.read_pickle(os.path.join(data_path, "train_val.pkl"))

        if split_type == "default":
            split_method = bm_split_names[self.name][dataset]
        else:
            split_method = split_type

        frac = [0.875, 0.125, 0.0]

        if split_method == "scaffold":
            out = create_scaffold_split(train_val,
                                        seed,
                                        frac=frac,
                                        entity="Drug")
        elif split_method == "random":
            out = create_fold(train_val, seed, frac=frac)
        elif split_method == "combination":
            out = create_combination_split(train_val, seed, frac=frac)
        elif split_method == "group":
            out = create_group_split(train_val,
                                     seed,
                                     holdout_frac=0.2,
                                     group_column="Year")
        else:
            raise NotImplementedError
        return out["train"], out["valid"]

    def get(self, benchmark):
        """get individual benchmark

        Args:
            benchmark (str): benchmark name

        Returns:
            dict: a dictionary of train_val, test dataframes and normalized name of the benchmark
        """
        dataset = fuzzy_search(benchmark, self.dataset_names)
        data_path = os.path.join(self.path, dataset)
        if self.file_format == "csv":
            train = pd.read_csv(os.path.join(data_path, "train_val.csv"))
            test = pd.read_csv(os.path.join(data_path, "test.csv"))
        elif self.file_format == "pkl":
            train = pd.read_pickle(os.path.join(data_path, "train_val.pkl"))
            test = pd.read_pickle(os.path.join(data_path, "test.pkl"))

        return {"train_val": train, "test": test, "name": dataset}

    def evaluate(self, pred, testing=True, benchmark=None, save_dict=True):
        """automatic evaluation function

        Args:
            pred (dict): a dictionary of benchmark name as the key and prediction array as the value
            testing (bool, optional): evaluate using testing set mode or validation set mode
            benchmark (str, optional): name of the benchmark
            save_dict (bool, optional): whether or not to save the evaluation result

        Returns:
            dict: a dictionary with key the benchmark name and value a dictionary of metrics to metric value

        Raises:
            ValueError: benchmark name not found
        """
        if testing:
            # test set evaluation
            metric_dict = bm_metric_names[self.name]
            out = {}
            for data_name, pred_ in pred.items():
                data_name = fuzzy_search(data_name, self.dataset_names)
                data_path = os.path.join(self.path, data_name)
                if self.file_format == "csv":
                    test = pd.read_csv(os.path.join(data_path, "test.csv"))
                elif self.file_format == "pkl":
                    test = pd.read_pickle(os.path.join(data_path, "test.pkl"))
                y = test.Y.values
                evaluator = eval("Evaluator(name = '" + metric_dict[data_name] +
                                 "')")
                out[data_name] = {
                    metric_dict[data_name]: round(evaluator(y, pred_), 3)
                }

                # If reporting accuracy across target classes
                if "target_class" in test.columns:
                    test["pred"] = pred_
                    for c in test["target_class"].unique():
                        data_name_subset = data_name + "_" + c
                        test_subset = test[test["target_class"] == c]
                        y_subset = test_subset.Y.values
                        pred_subset = test_subset.pred.values

                        evaluator = eval("Evaluator(name = '" +
                                         metric_dict[data_name_subset] + "')")
                        out[data_name_subset] = {
                            metric_dict[data_name_subset]:
                                round(evaluator(y_subset, pred_subset), 3)
                        }
            return out
        else:
            # validation set evaluation
            if benchmark is None:
                raise ValueError(
                    "Please specify the benchmark name for us to retrieve the standard metric!"
                )
            data_name = fuzzy_search(benchmark, self.dataset_names)
            metric_dict = bm_metric_names[self.name]
            evaluator = eval("Evaluator(name = '" + metric_dict[data_name] +
                             "')")
            return {metric_dict[data_name]: round(evaluator(true, pred), 3)}

    def evaluate_many(self,
                      preds,
                      save_file_name=None,
                      results_individual=None):
        """
        This function returns the data in a format needed to submit to the Leaderboard

        Args:
            preds (list of dict): list of dictionary of predictions, each item is the input to the evaluate function.
            save_file_name (str, optional): file name to save the result
            results_individual (list of dictionary, optional): if you already have results generated for each run, simply input here so that this function won't call the evaluation function again

        Returns:
            dict: a dictionary where key is the benchmark name and value is another dictionary where the key is the metric name and value is a list [mean, std].
        """
        min_requirement = 5

        if len(preds) < min_requirement:
            return ValueError("Must have predictions from at least " +
                              str(min_requirement) +
                              " runs for leaderboard submission")
        if results_individual is None:
            individual_results = []
            for pred in preds:
                retval = self.evaluate(pred)
                individual_results.append(retval)
        else:
            individual_results = results_individual

        given_dataset_names = list(individual_results[0].keys())
        aggregated_results = {}
        for dataset_name in given_dataset_names:
            my_results = []
            for individual_result in individual_results:
                my_result = list(individual_result[dataset_name].values())[0]
                my_results.append(my_result)
            u = np.mean(my_results)
            std = np.std(my_results)
            aggregated_results[dataset_name] = [round(u, 3), round(std, 3)]
        return aggregated_results
