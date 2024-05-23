import unittest
import shutil
import numpy as np
import random
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from tdc.benchmark_group import admet_group, scdti_group, counterfactual_group, geneperturb_group


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
        print("got test data")
        group = counterfactual_group.CounterfactualGroup()  # is drug
        assert group.is_drug
        assert set(group.dataset_names) == set(
            scperturb_datasets
        ), "loaded datasets should be scperturb drug, but were {} vs correct: {}".format(
            group.dataset_names, scperturb_datasets)
        print("getting test data via dataloader")
        ct = len(test_data.get_data())
        print("getting splits")
        train, val = group.get_train_valid_split(remove_unseen=False)
        test = group.get_test()
        print("got splits; checking counts")
        trainct = sum(len(x) for _, x in train.items())
        valct = sum(len(x) for _, x in val.items())
        testct = sum(len(x) for _, x in test.items())
        controlct = sum(len(x) for _, x in group.split["control"].items())
        adjct = group.split["adj"]
        totalct = trainct + valct + testct + controlct + adjct
        assert ct == totalct, "counts between original data and the 3 splits should match: original {} vs splits {}".format(
            ct, totalct)
        # basic test on perfect score
        print("benchmark - generating identical test set")
        tst = test_data.get_split(remove_unseen=False)
        tstdict = {}
        for line, splits in tst.items():
            tstdict[line] = splits["test"]
        print("benchmark - running evaluate")
        r2 = group.evaluate(tstdict)
        assert r2 == 1, "comparing test to itself should have perfect R^2 score, was {}".format(
            r2)
        print("done")
        # now just check we can load sc perturb gene correctly
        print("benchmark - basic load test on the gene benchmark")
        group_gene = counterfactual_group.CounterfactualGroup(is_drug=False)
        assert not group_gene.is_drug
        assert set(group_gene.dataset_names) == set(scperturb_gene_datasets)

    @unittest.skip(
        "counterfactual test is taking up too much memory"
    )  #FIXME: please run if making changes to counterfactual benchmark or core code.
    def test_gene_perturb(self):
        group = geneperturb_group.GenePerturbGroup()
        group.get_train_valid_split()
        group.get_test()

    def test_proteinpeptide(self):
        from tdc.benchmark_group.protein_peptide_group import ProteinPeptideGroup
        from tdc.multi_pred.proteinpeptide import ProteinPeptide
        from sklearn.model_selection import train_test_split
        group = ProteinPeptideGroup()
        test = group.get_test()
        assert test is not None and len(test) > 0
        dl = ProteinPeptide(name="brown_mdm2_ace2_12ca5")
        df = dl.get_data()
        for idx, e in enumerate(df["Y"]):
            if e != "Putative binder":
                df["Y"][idx] = "1"
            else:
                df["Y"][idx] = "0"
        # raise Exception("unique", )
        # Split the data while stratifying
        _, _, _, y_test = train_test_split(
            df.drop('Y', axis=1),  # features
            df['Y'],  # labels
            test_size=0.9,  # 90% of the data goes to the test set
            random_state=42,  # for reproducibility
            stratify=df[
                'Y']  # stratify by the label column to ensure even distribution
        )
        res = group.evaluate(y_test)
        assert res[-1] == 1 and res[-2] == 1, res

    def test_tcrepitope(self):
        from tdc.benchmark_group.tcrepitope_group import TCREpitopeGroup
        from tdc.resource.dataloader import DataLoader
        data = DataLoader("tchard")
        tst = data.get_split()["test"]
        group = TCREpitopeGroup()
        res = group.evaluate(tst)
        assert res == 1


if __name__ == "__main__":
    unittest.main()
