import unittest
import shutil
import numpy as np
import random
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tdc.benchmark_group import admet_group, scdti_group


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
        group = scdti_group.SCDTIGroup()
        train, val = group.get_train_valid_split()
        assert len(val) == 0  # this benchmark has no validation set
        # test simple preds
        y_true = group.get_test()["Y"]
        results = group.evaluate(y_true)
        assert results[-1] == 1.0  # should be perfect F1 score
        zero_pred = [0] * len(y_true)
        results = group.evaluate(zero_pred)
        assert results[-1] != 1.0  # should not be perfect F1 score


if __name__ == "__main__":
    unittest.main()
