import unittest
import shutil
import numpy as np
import random
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tdc.benchmark_group import admet_group, scdti_group, counterfactual_group


def is_classification(values):
    value_set = set(values)
    if len(value_set) == 2:
        return True
    return False


class TestBenchmarkGroup(unittest.TestCase):

    def setUp(self):
        self.group = admet_group(path="data/")

    def tearDown(self) -> None:
        shutil.rmtree("data", ignore_errors=True)

    def test_ADME_mean_prediction(self):
        predictions = {}
        num_datasets = 0
        for my_group in self.group:
            num_datasets += 1
            name = my_group["name"]
            test = my_group["test"]
            mean_val = np.mean(test["Y"])
            predictions[name] = [mean_val] * len(test)

        results = self.group.evaluate(predictions)
        self.assertEqual(num_datasets, len(results))
        for my_group in self.group:
            self.assertTrue(my_group["name"] in results)

    def test_ADME_evaluate_many(self):
        prediction_list = []
        for random_seed in range(5):
            predictions = {}
            for my_group in self.group:
                name = my_group["name"]
                test = my_group["test"]
                predictions[name] = test["Y"]
            prediction_list.append(predictions)

        results = self.group.evaluate_many(prediction_list)
        for ds_name, metrics in results.items():
            self.assertEqual(len(metrics), 2)
            u, std = metrics
            self.assertTrue(u
                            in (1,
                                0))  # A perfect score for all metrics is 1 or 0
            self.assertEqual(0, std)

        for my_group in self.group:
            self.assertTrue(my_group["name"] in results)

    def test_SCDTI_benchmark(self):
        from tdc.resource.dataloader import DataLoader

        data = DataLoader(name="opentargets_dti")
        group = scdti_group.SCDTIGroup()
        train, val = group.get_train_valid_split()
        assert len(val) == 0  # this benchmark has no validation set
        # test simple preds
        y_true = group.get_test()["Y"]
        results = group.evaluate(y_true)
        assert results[-1] == 1.0  # should be perfect F1 score
        # assert it matches the opentargets official test scores
        tst = data.get_split()["test"]["Y"]
        results = group.evaluate(tst)
        assert results[-1] == 1.0
        zero_pred = [0] * len(y_true)
        results = group.evaluate(zero_pred)
        assert results[-1] != 1.0  # should not be perfect F1 score
        many_results = group.evaluate_many([y_true] * 5)
        assert "f1" in many_results
        assert len(many_results["f1"]
                  ) == 2  # should include mean and standard deviation

    @unittest.skip(
        "counterfactual test is taking up too much memory"
    )  #FIXME: please run if making changes to counterfactual benchmark or core code.
    def test_counterfactual(self):
        from tdc.multi_pred.perturboutcome import PerturbOutcome
        from tdc.dataset_configs.config_map import scperturb_datasets, scperturb_gene_datasets

        test_data = PerturbOutcome("scperturb_drug_AissaBenevolenskaya2021")
        group = counterfactual_group.CounterfactualGroup()  # is drug
        assert group.is_drug
        assert set(group.dataset_names) == set(
            scperturb_datasets
        ), "loaded datasets should be scperturb drug, but were {} vs correct: {}".format(
            group.dataset_names, scperturb_datasets)
        ct = len(test_data.get_data())
        train, val = group.get_train_valid_split()
        test = group.get_test()
        control = group.split["control"]
        testct = len(train) + len(val) + len(test) + len(control)
        assert ct == testct, "counts between original data and the 3 splits should match: original {} vs splits {}".format(
            ct, testct)
        # basic test on perfect score
        tst = test_data.get_split()["test"]
        r2 = group.evaluate(tst)
        assert r2 == 1, "comparing test to itself should have perfect R^2 score, was {}".format(
            r2)
        # now just check we can load sc perturb gene correctly
        group_gene = counterfactual_group.CounterfactualGroup(is_drug=False)
        assert not group_gene.is_drug
        assert set(group_gene.dataset_names) == set(scperturb_gene_datasets)


if __name__ == "__main__":
    unittest.main()
