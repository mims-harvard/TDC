import unittest
import shutil
import numpy as np
import random
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '../..')))
from tdc.benchmark import BenchmarkGroup

def is_classification(values):
    value_set = set(values)
    if len(value_set) == 2:
        return True
    return False


class TestBenchmarkGroup(unittest.TestCase):
    
    def setUp(self):
        self.group = BenchmarkGroup(name='ADMET_Group', path='data/')

    def tearDown(self) -> None:
        shutil.rmtree('data', ignore_errors=True)

    def test_ADME_mean_prediction(self):
        predictions = {}
        num_datasets = 0
        for my_group in self.group:
            num_datasets += 1
            name = my_group['name']
            test = my_group['test']
            mean_val = np.mean(test['Y'])
            predictions[name] = [mean_val] * len(test)

        results = self.group.evaluate(predictions)
        self.assertEqual(num_datasets, len(results))
        for my_group in self.group:
            self.assertTrue(my_group['name'] in results)

    def test_ADME_evaluate_many(self):
        prediction_list = []
        for random_seed in range(5):
            predictions = {}
            for my_group in self.group:
                name = my_group['name']
                test = my_group['test']
                predictions[name] = test['Y']
            prediction_list.append(predictions)

        results = self.group.evaluate_many(prediction_list)
        for ds_name, metrics in results.items():
            self.assertEqual(len(metrics), 2)
            u, std = metrics
            self.assertTrue(u in (1, 0))  # A perfect score for all metrics is 1 or 0
            self.assertEqual(0, std)

        for my_group in self.group:
            self.assertTrue(my_group['name'] in results)
