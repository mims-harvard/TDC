import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import unittest

from pandas import DataFrame
# from pyarrow import SparseCOOTensor
from tdc.resource import cellxgene_census


class TestResources(unittest.TestCase):
    pass


class TestCellXGene(unittest.TestCase):

    def setUp(self):
        self.resource = cellxgene_census.CensusResource()
        self.gene_value_filter = "feature_id in ['ENSG00000161798', 'ENSG00000188229']"
        self.gene_column_names = ["feature_name", "feature_length"]
        self.cell_value_filter = "tissue == 'brain' and sex == 'male'"
        self.cell_column_names = ["assay", "cell_type", "tissue"]

    def test_get_cell_metadata(self):
        obsdf = self.resource.get_cell_metadata(
            value_filter=self.cell_value_filter,
            column_names=self.cell_column_names,
            fmt="pandas")
        assert isinstance(obsdf, DataFrame)

    def test_get_gene_metadata(self):
        varpyarrow = self.resource.get_gene_metadata(
            value_filter=self.gene_value_filter,
            column_names=self.gene_column_names,
            fmt="pyarrow",
            measurement_name="RNA")
        print(varpyarrow)
        # assert isinstance(varpyarrow, SparseCOOTensor)

    def test_get_measurement_matrix(self):
        X = self.resource.query_measurement_matrix(
            measurement_name="RNA",
            fmt="pyarrow",
            value_adjustment="raw",
            value_filter="tissue == 'brain' and sex == 'male'")
        next(X)[:3]

    def test_get_feature_dataset_presence_matrix(self):
        FMslice = self.resource.get_feature_dataset_presence_matrix(
            upper=5,
            lower=0,
            measurement_name="RNA",
            fmt="pyarrow",
            todense=False)
        print("f", FMslice)


if __name__ == "__main__":
    unittest.main()
