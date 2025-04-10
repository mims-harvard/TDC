import unittest
import shutil
import numpy as np

from tdc.benchmark_group import admet_group, scdti_group, counterfactual_group, geneperturb_group
from tdc.benchmark_group.protein_peptide_group import ProteinPeptideGroup
from tdc.multi_pred.proteinpeptide import ProteinPeptide
from sklearn.model_selection import train_test_split
from tdc.multi_pred.perturboutcome import PerturbOutcome
from tdc.dataset_configs.config_map import scperturb_datasets, scperturb_gene_datasets
from tdc.benchmark_group.tcrepitope_group import TCREpitopeGroup
from tdc.resource.dataloader import DataLoader


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
        train_val = group.get_train_valid_split()
        assert "train" in train_val, "no training set"
        assert "val" in train_val, "no validation set"
        assert len(train_val["train"]) > 0, "no entries in training set"
        tst = group.get_test()["test"]
        tst["preds"] = tst["y"]  # switch predictions to ground truth
        results = group.evaluate(tst)
        assert "IBD" in results, "missing ibd from diseases. got {}".format(
            results.keys())
        assert "RA" in results, "missing ra from diseases. got {}".format(
            results.keys())
        assert results["IBD"] == results[
            "RA"], "both should be perfect scores but got IBD {} vs RA {}".format(
                results["IBD"], results["RA"])  # both should be perfect scores
        assert results["IBD"] - 1.0 < 0.000001  # should be a perfect score
        many_results = group.evaluate_many([tst] * 5)
        assert "IBD" in many_results, "missing ibd from diseases in evaluate many. got {}".format(
            many_results.keys())
        assert "RA" in many_results, "missing ra from diseases in evaluate many. got {}".format(
            many_results.keys())
        assert len(many_results["IBD"]) == len(
            many_results["RA"]
        ), "both diseases should include mean and standard deviation"
        assert len(many_results["IBD"]
                  ) == 2, "results should include mean and standard deviation"
        assert many_results["IBD"][
            0] - 1.0 < 0.000001, "should get perfect score"

    @unittest.skip(
        "counterfactual test is taking up too much memory"
    )  #FIXME: please run if making changes to counterfactual benchmark or core code.
    def test_counterfactual(self):
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

    @unittest.skip(
        "mygene dependency removal")  #FIXME: separate into conda-only tests
    def test_proteinpeptide(self):
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
        data = DataLoader("tchard")
        tst = data.get_split()["test"]
        group = TCREpitopeGroup()
        res = group.evaluate(tst)
        assert res == 1
