import cellxgene_census
from pandas import concat
import tiledbsoma

from tdc import base_dataset
"""
    
Are we only supporting memory-efficient queries?
https://chanzuckerberg.github.io/cellxgene-census/cellxgene_census_docsite_quick_start.html#memory-efficient-queries


"""


class CXGDataLoader(base_dataset.DataLoader):

    def __init__(self,
                 num_slices=None,
                 census_version="2023-12-15",
                 dataset="census_data",
                 organism="homo_sapiens",
                 measurement_name="RNA",
                 value_filter="",
                 column_names=None):
        if column_names is None:
            raise ValueError("column_names is required for this loader")
        self.column_names = column_names
        num_slices = num_slices if num_slices is not None else 1
        self.num_slices = num_slices
        self.df = None
        self.fetch_data(census_version, dataset, organism, measurement_name,
                        value_filter)

    def fetch_data(self, census_version, dataset, organism, measurement_name,
                   value_filter):
        """TODO: docs
        outputs a dataframe with specified query params on census data SOMA collection object
        """
        if self.column_names is None:
            raise ValueError(
                "Column names must be provided to CXGDataLoader class")

        with cellxgene_census.open_soma(
                census_version=census_version) as census:
            # Reads SOMADataFrame as a slice
            cell_metadata = census[dataset][organism].obs.read(
                value_filter=value_filter, column_names=self.column_names)
            self.df = cell_metadata.concat().to_pandas()
            # TODO: not latency on memory-efficient queries is poor...
            # organismCollection = census[dataset][organism]
            # query = organismCollection.axis_query(
            #     measurement_name = measurement_name,
            #     obs_query = tiledbsoma.AxisQuery(
            #         value_filter = value_filter
            #     )
            # )
            # it = query.X("raw").tables()
            # dfs =[]
            # for  _ in range(self.num_slices):
            #     slice = next (it)
            #     df_slice = slice.to_pandas()
            #     dfs.append(df_slice)
            # self.df = concat(dfs)

    def get_dataframe(self):
        if self.df is None:
            raise Exception(
                "Haven't instantiated a DataFrame yet. You can call self.fetch_data first."
            )
        return self.df


if __name__ == "__main__":
    # TODO: tmp, run testing suite when this file is called as main
    loader = CXGDataLoader(value_filter="tissue == 'brain' and sex == 'male'",
                           column_names=["assay", "cell_type", "tissue"])
    df = loader.get_dataframe()
    print(df.head())
