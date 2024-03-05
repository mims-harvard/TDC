# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import pandas as pd
import numpy as np
import os, sys, json
import warnings

warnings.filterwarnings("ignore")

from .base_group import BenchmarkGroup
from ..utils import bm_group_load, print_sys, fuzzy_search
from ..metadata import (
    get_task2category,
    bm_metric_names,
    benchmark_names,
    bm_split_names,
    docking_target_info,
)
from ..evaluator import Evaluator


class docking_group(BenchmarkGroup):
    """Create a docking group benchmark loader.

    Args:
        path (str, optional): the folder path to save/load the benchmarks.
        pyscreener_path (str, optional): the path to pyscreener repository in order to call docking scores.
        num_workers (int, optional): number of workers to parallelize dockings
        num_cpus (int, optional): number of CPUs assigned to docking
        num_max_call (int, optional): maximum number of oracle calls


    """

    def __init__(self,
                 path="./data",
                 num_workers=None,
                 num_cpus=None,
                 num_max_call=5000):
        """Create a docking group benchmark loader.

        Raises:
            ValueError: missing path to pyscreener.
        """
        super().__init__(name="Docking_Group", path=path, file_format="oracle")

        # if pyscreener_path is not None:
        # 	self.pyscreener_path = pyscreener_path
        # else:
        # 	raise ValueError("Please specify pyscreener_path!")

        if (num_workers is None) and (num_cpus is None):
            ## automatic selections
            cpu_total = os.cpu_count()
            if cpu_total > 1:
                num_cpus = 2
            else:
                num_cpus = 1
            num_workers = int(cpu_total / num_cpus)

        self.num_workers = num_workers
        self.num_cpus = num_cpus
        self.num_max_call = num_max_call
        from ..oracles import Oracle

    def __iter__(self):
        """iterate docking targets

        Returns:
            docking_group: the docking group class itself
        """
        self.index = 0
        self.num_datasets = len(self.dataset_names)
        return self

    def __next__(self):
        """retrieve the next benchmark

        Returns:
            dict: a dictionary of oracle function, molecule library dataset, and the name of docking target

        Raises:
            StopIteration: stop when all benchmarks are obtained.
        """
        if self.index < self.num_datasets:
            dataset = self.dataset_names[self.index]
            print_sys("--- " + dataset + " ---")

            data_path = os.path.join(self.path, dataset)
            if not os.path.exists(data_path):
                os.mkdir(data_path)

            target_pdb_file = os.path.join(self.path, dataset + ".pdb")
            self.index += 1

            from ..oracles import Oracle

            # oracle = Oracle(name = "Docking_Score", software="vina",
            # 	pyscreener_path = self.pyscreener_path,
            # 	receptors=[target_pdb_file],
            # 	center=docking_target_info[dataset]['center'], size=docking_target_info[dataset]['size'],
            # 	buffer=10, path=data_path, num_worker=self.num_workers, ncpu=self.num_cpus, num_max_call = self.num_max_call)
            oracle = Oracle(
                name="Docking_Score",
                pyscreener_path=self.pyscreener_path,
                receptors=[target_pdb_file],
                center=docking_target_info[dataset]["center"],
                size=docking_target_info[dataset]["size"],
                buffer=10,
                path=data_path,
                num_worker=self.num_workers,
                ncpu=self.num_cpus,
                num_max_call=self.num_max_call,
            )
            data = pd.read_csv(os.path.join(self.path, "zinc.tab"), sep="\t")
            return {"oracle": oracle, "data": data, "name": dataset}
        else:
            raise StopIteration

    def get_train_valid_split(self, seed, benchmark, split_type="default"):
        """no split for docking group

        Raises:
            ValueError: no split for docking group
        """
        raise ValueError(
            "Docking molecule generation does not have the concept of training/testing split! Checkout the usage in tdcommons.ai !"
        )

    def get(self, benchmark, num_max_call=5000):
        """retrieve one benchmark given benchmark name (docking target)

        Args:
            benchmark (str): the name of the benchmark
            num_max_call (int, optional): maximum of oracle calls

        Returns:
            dict: a dictionary of oracle function, molecule library dataset, and the name of docking target
        """
        dataset = fuzzy_search(benchmark, self.dataset_names)
        data_path = os.path.join(self.path, dataset)
        target_pdbqt_file = os.path.join(self.path, dataset + ".pdbqt")

        from ..oracles import Oracle

        # oracle = Oracle(name = "Docking_Score", software="vina",
        # 	pyscreener_path = self.pyscreener_path,
        # 	receptors=[target_pdb_file],
        # 	center=docking_target_info[dataset]['center'], size=docking_target_info[dataset]['size'],
        # 	buffer=10, path=data_path, num_worker=self.num_workers, ncpu=self.num_cpus, num_max_call = num_max_call)
        # oracle = Oracle(name = "Docking_Score",
        # 	receptor_pdbqt_file=target_pdbqt_file,
        # 	center=docking_target_info[dataset]['center'],
        # 	box_size=docking_target_info[dataset]['size'],
        # 	num_max_call = num_max_call)
        oracle = Oracle(name="3pbl_docking")
        data = pd.read_csv(os.path.join(self.path, "zinc.tab"), sep="\t")
        return {"oracle": oracle, "data": data, "name": dataset}

    def evaluate(self,
                 pred,
                 true=None,
                 benchmark=None,
                 m1_api=None,
                 save_dict=True):
        """Summary

        Args:
            pred (dict): a nested dictionary, where the first level key is the docking target, the value is another dictionary where the key is the maximum oracle calls, and value can have two options. One, a dictionary of SMILES paired up with the docking scores and Second, a list of SMILES strings, where the function will generate the docking scores automatically.
            benchmark (str, optional): name of the benchmark docking target.
            m1_api (str, optional): API token of Molecule.One. This is to use M1 service to generate synthesis score.
            save_dict (bool, optional): whether or not to save the results.

        Returns:
            dict: result with all realistic metrics generated

        Raises:
            ValueError: Description
        """
        results_all = {}

        for data_name, pred_all in pred.items():
            results_max_call = {}
            for num_max_call, pred_ in pred_all.items():

                results = {}

                recalc = False

                if isinstance(pred_, dict):
                    print_sys(
                        "The input is a dictionary, expected to have SMILES string as key and docking score as value!"
                    )
                    docking_scores = pred_
                    pred_ = list(pred_.keys())
                elif isinstance(pred_, list):
                    recalc = True
                    print_sys(
                        "The input is a list, docking score will be computed! If you already have the docking scores, please make the list as a dictionary with SMILES string as key and docking score as value"
                    )
                else:
                    raise ValueError(
                        "The input prediction must be a dictionary with SMILES and their docking scores or a list of SMILES!"
                    )
                ## pred is a list of smiles strings or a dictionary of smiles strings if docking scores are already calculated...
                if len(pred_) != 100:
                    raise ValueError(
                        "The expected output is a list/dictionary of top 100 molecules!"
                    )

                if recalc:
                    dataset = fuzzy_search(data_name, self.dataset_names)

                    # docking scores for the top K smiles (K <= 100)
                    target_pdb_file = os.path.join(self.path, dataset + ".pdb")
                    from ..oracles import Oracle

                    data_path = os.path.join(self.path, dataset)
                    oracle = Oracle(
                        name="Docking_Score",
                        software="vina",
                        pyscreener_path=self.pyscreener_path,
                        receptors=[target_pdb_file],
                        center=docking_target_info[dataset]["center"],
                        size=docking_target_info[dataset]["size"],
                        buffer=10,
                        path=data_path,
                        num_worker=self.num_workers,
                        ncpu=self.num_cpus,
                        num_max_call=10000,
                    )

                    docking_scores = oracle(pred_)
                print_sys("---- Calculating average docking scores ----")
                if len(
                        np.where(np.array(list(docking_scores.values())) > 0)
                    [0]) > 0.7:
                    ## check if the scores are all positive.. if so, make them all negative
                    docking_scores = {j: -k for j, k in docking_scores.items()}
                if save_dict:
                    results["docking_scores_dict"] = docking_scores
                values = np.array(list(docking_scores.values()))
                results["top100"] = np.mean(values)
                results["top10"] = np.mean(sorted(values)[:10])
                results["top1"] = min(values)

                # if m1_api is None:
                # 	print_sys('Ignoring M1 Synthesizability Evaluations. You can still submit your results without m1 score. Although for the submission, we encourage inclusion of m1 scores. To opt-in, set the m1_api to the token obtained via: https://tdcommons.ai/functions/oracles/#moleculeone')
                # else:
                # 	print_sys("---- Calculating molecule.one synthesizability score ----")
                # 	from ..oracles import Oracle
                # 	m1 = Oracle(name = 'Molecule One Synthesis', api_token = m1_api)
                # 	import heapq
                # 	from operator import itemgetter
                # 	top10_docking_smiles = list(dict(heapq.nsmallest(10, docking_scores.items(), key=itemgetter(1))).keys())
                # 	m1_scores = m1(top10_docking_smiles)
                # 	scores_array = list(m1_scores.values())
                # 	scores_array = np.array([float(i) for i in scores_array])
                # 	scores_array[np.where(scores_array == -1.0)[0]] = 10 # m1 score errors are usually large complex molecules
                # 	if save_dict:
                # 		results['m1_dict'] = m1_scores
                # 	results['m1'] = np.mean(scores_array)

                print_sys("---- Calculating synthetic accessibility score ----")
                from ..oracles import Oracle

                sa = Oracle(name="SA")
                scores_array = sa(pred_)
                if save_dict:
                    results["sa_dict"] = scores_array
                results["sa"] = np.mean(scores_array)

                print_sys("---- Calculating molecular filters scores ----")
                from ..chem_utils.oracle.filter import MolFilter

                ## follow guacamol
                filters = MolFilter(
                    filters=["PAINS", "SureChEMBL", "Glaxo"],
                    property_filters_flag=False,
                )
                pred_filter = filters(pred_)
                if save_dict:
                    results["pass_list"] = pred_filter
                results["%pass"] = float(len(pred_filter)) / 100
                results["top1_%pass"] = min(
                    [docking_scores[i] for i in pred_filter])
                print_sys("---- Calculating diversity ----")
                from ..evaluator import Evaluator

                evaluator = Evaluator(name="Diversity")
                score = evaluator(pred_)
                results["diversity"] = score
                print_sys("---- Calculating novelty ----")
                evaluator = Evaluator(name="Novelty")
                training = pd.read_csv(os.path.join(self.path, "zinc.tab"),
                                       sep="\t")
                score = evaluator(pred_, training.smiles.values)
                results["novelty"] = score
                results["top smiles"] = [
                    i[0]
                    for i in sorted(docking_scores.items(), key=lambda x: x[1])
                ]
                results_max_call[num_max_call] = results
            results_all[data_name] = results_max_call
        return results_all

    def evaluate_many(self,
                      preds,
                      save_file_name=None,
                      m1_api=None,
                      results_individual=None):
        """evaluate many runs together and output submission ready pkl file.

        Args:
            preds (list): a list of pred across runs, where each follows the format of pred in 'evaluate' function.
            save_file_name (str, optional): the name of the file to save the result.
            m1_api (str, optional): m1 API token for molecule synthesis score.
            results_individual (list, optional): if you already have generated the result from the evaluate function for each run, simply put in a list and it will not regenerate the results.

        Returns:
            dict: the output result file.
        """
        min_requirement = 3
        if len(preds) < min_requirement:
            return ValueError("Must have predictions from at least " +
                              str(min_requirement) +
                              " runs for leaderboard submission")
        if results_individual is None:
            individual_results = []
            for pred in preds:
                retval = self.evaluate(pred, m1_api=m1_api)
                individual_results.append(retval)
        else:
            individual_results = results_individual

        metrics = [
            "top100",
            "top10",
            "top1",
            "diversity",
            "novelty",
            "%pass",
            "top1_%pass",
            "sa",
            "top smiles",
        ]
        num_folds = len(preds)

        results_agg = {}

        for target in list(individual_results[0].keys()):
            results_agg_target = {}
            for num_calls in individual_results[0][target].keys():
                results_agg_target_call = {}
                for metric in metrics:
                    if metric == "top smiles":
                        results_agg_target_call[metric] = np.unique(
                            np.array([
                                individual_results[fold][target][num_calls]
                                [metric] for fold in range(num_folds)
                            ]).reshape(-1)).tolist()
                    else:
                        res = [
                            individual_results[fold][target][num_calls][metric]
                            for fold in range(num_folds)
                        ]
                        results_agg_target_call[metric] = [
                            round(np.mean(res), 3),
                            round(np.std(res), 3),
                        ]
                results_agg_target[num_calls] = results_agg_target_call
            results_agg[target] = results_agg_target

        import pickle

        if save_file_name is None:
            save_file_name = "tdc_docking_result"
        with open(save_file_name + ".pkl", "wb") as f:
            pickle.dump(results_agg, f)
        return results_agg
