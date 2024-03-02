import cellxgene_census
from pandas import concat
import tiledbsoma


class CensusResource:

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
            print("reading data...")
            obs = census[dataset][organism].obs
            print("got var df")
            print("filtering")
            varread = obs.read(value_filter=value_filter, column_names = self.column_names)
            print("converting to pandas.. first pyarrow")
            cc = varread.concat()
            print("now to pandas")
            self.df = cc.to_pandas()
            print(self.df.head())
            print("now running var test")
            
            var = census[dataset][organism].ms[measurement_name].var
            self.df = var.read(column_names=["feature_name", "feature_reference"], value_filter="feature_id in ['ENSG00000161798', 'ENSG00000188229']").concat().to_pandas()
            print("printing var")
            print(self.df.head())
            # TODO: gene / var columns are var: 'soma_joinid', 'feature_id', 'feature_name', 'feature_length'

            print("now testing queries on data matrices (X)")
            n_obs = len(obs)
            n_var =len(var)
            X = census[dataset][organism].ms[measurement_name].X["raw"]
            slc = X.read((slice(0, 5),)).coos((n_obs,n_var)) # need bounding boxes
            self.df = slc.concat()
            print(self.df)
            print("prinnting  X")
            print(self.df.to_scipy().todense())
            
            print("now testing feature dataset presence matrix")
            fMatrix = census[dataset][organism].ms[measurement_name]["feature_dataset_presence_matrix"]
            slc = fMatrix.read((slice(0, 5),)).coos((n_obs,n_var)) # need bounding boxes
            self.df = slc.concat()
            print(self.df)
            print("printing ftp matrix")
            print(self.df.to_scipy().todense())
            
            print("can we do full read on X?")
            bded = X.read().coos((n_obs, n_var)) # still need bounding boxes
            print("can get the sparse array coos()")
            print("we cannot get pyarrow")
            # print("pyarrow")
            # bded.concat()
            print("yes we can")
            
            # X = census[dataset][organism].ms[measurement_name].X["raw"]
            # sparse_array = X.read()
            # print("spare array...")
            # print(sparse_array)
            # # TODO: tmp
            # print("converting to pandas")
            # print("first to pyarrow")
            # self.df = sparse_array.coos().concat()
            # print("done")
            # print("now pandas")
            # self.df = self.df.to_pandas()
            # print("done")
            
            
            # .read(
            #     value_filter=value_filter, column_names=self.column_names)
            # self.df = cell_metadata.concat().to_pandas()
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

    def get_data(self, type="df"):
        if type == "df":
            return self.get_dataframe()
        elif type == "pyarrow":
            raise Exception("PyArrow format not supported by TDC yet.")
        else:
            raise Exception("Type must be set to df or pyarrow")


if __name__ == "__main__":
    # TODO: tmp, run testing suite when this file is called as main
    print("initializing object")
    loader = CensusResource(value_filter="tissue == 'brain' and sex == 'male'",
                           column_names=["assay", "cell_type", "tissue"])
    print("getting")
    df = loader.get_data()
    print("getting head()")
    # print(df.head())
    print("no dense")
    print(df.to_scipy())
    print("done!")
