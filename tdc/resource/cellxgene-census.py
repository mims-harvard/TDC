# TODO: tmp fix
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
# TODO: find better fix or encode in environment / docker ^^^
import cellxgene_census
from functools import wraps
from pandas import concat
import tiledbsoma


class CensusResource:
    
    _CENSUS_DATA = "census_data"
    _FEATURE_PRESENCE = "feature_dataset_presence_matrix"
    _LATEST_CENSUS = "2023-12-15"
    _HUMAN = "homo_sapiens"

    class decorators:
        @classmethod
        def check_dataset_is_census_data(cls,func):
            # @wraps(func)
            def check(*args, **kwargs):
                self = args[0]
                if self.dataset != self._CENSUS_DATA:
                    raise ValueError("This function requires the '{}' dataset".format(self._CENSUS_DATA))
                return func(*args, **kwargs)
            return check

    def __init__(self,
                 census_version=None,
                 dataset=None,
                 organism=None
    ):
        """Initialize the Census Resource.
        
        Args:
            census_version (str): The date of the census data release in YYYY- 
            TODO: complete
        """
        self.census_version = census_version if census_version is not None else self._LATEST_CENSUS
        self.dataset = dataset if dataset is not None else self._CENSUS_DATA
        self.organism = organism if organism is not None else self._HUMAN

    def fmt_cellxgene_data(self, tiledb_ptr, fmt=None):
        if fmt is None:
            raise Exception("format not provided to fmt_cellxgene_data(), please provide fmt variable")
        elif fmt == "pandas":
            return tiledb_ptr.concat().to_pandas()
        elif fmt == "pyarrow":
            return tiledb_ptr.concat()
        elif fmt == "scipy":
            return tiledb_ptr.concat().to_scipy()
        else:
            raise Exception("fmt not in [pandas, pyarrow, scipy] for fmt_cellxgene_data()")
    
    @decorators.check_dataset_is_census_data
    def get_cell_metadata(self, value_filter=None, column_names=None, fmt=None):
        """Get the cell metadata (obs) data from the Census API"""
        if value_filter is None:
            raise Exception("No value filter was provided, dataset is too large to fit in memory. \
                            Memory-Efficient queries are not supported yet.")
        fmt = fmt if fmt is not None else "pandas"
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            obs = census[self.dataset][self.organism].obs
            obsread = None
            if column_names:
                obsread = obs.read(value_filter=value_filter, column_names=column_names)
            else:
                obsread = obs.read(value_filter=value_filter)
            return self.fmt_cellxgene_data(obsread, fmt)
        
    @decorators.check_dataset_is_census_data
    def get_gene_metadata(self, value_filter=None, column_names=None, measurement_name=None, fmt=None):
        """Get the gene metadata (var) data from the Census API"""
        if value_filter is None:
            raise Exception("No value filter was provided, dataset is too large to fit in memory. \
                Memory-Efficient queries are not supported yet.")
        elif measurement_name is None:
            raise ValueError("measurment_name must be provided , i.e. 'RNA'")
        fmt = fmt if fmt is not None else "pandas"
        with cellxgene_census.open_soma(
            census_version=self.census_version
        ) as census:
            var =  census[self.dataset][self.organism].ms[measurement_name].var
            varread = None
            if column_names:
                varread = var.read(value_filter=value_filter, column_names=column_names)
            else:
                varread = var.read(value_filter=value_filter)
            return self.fmt_cellxgene_data(varread, fmt)
        
    @decorators.check_dataset_is_census_data
    def get_measurement_matrix(self, upper=None, lower=None, value_adjustment=None, measurement_name=None, fmt=None, todense=None):
        """Count matrix for an input measurement by slice

        Args:
            upper (_type_, optional): _description_. Defaults to None.
            lower (_type_, optional): _description_. Defaults to None.
            value_adjustment (_type_, optional): _description_. Defaults to None.
            measurement_name (_type_, optional): _description_. Defaults to None.

        Raises:
            Exception: _description_
            Exception: _description_
        """
        if upper is None or lower is None:
            raise Exception("No upper and/or lower bound for slicing was provided. Dataset is too large to fit in memory. \
                Memory-Efficient queries are not supported yet.")
        elif measurement_name is None:
            raise Exception("measurement_name was not provided.")
        elif fmt is not None and fmt not in ["scipy", "pyarrow"]:
            raise ValueError("measurement_matrix only supports 'scipy' or 'pyarrow' format")
        value_adjustment = value_adjustment if value_adjustment is not None else "raw"
        todense = todense if todense is not None else False
        fmt = fmt if fmt is not None else "scipy"
        if todense and fmt != "scipy":
            raise ValueError("dense representation only available in scipy format")
        with cellxgene_census.open_soma(
            census_version=self.census_version
        ) as census:
            n_obs = len(census[self.dataset][self.organism].obs)
            n_var = len(census[self.dataset][self.organism].ms[measurement_name].var)
            X = census[self.dataset][self.organism].ms[measurement_name].X[value_adjustment]
            slc = X.read((slice(lower, upper),)).coos((n_obs, n_var))
            out = self.fmt_cellxgene_data(slc, fmt)
            return out if not todense else out.todense()
    
    @decorators.check_dataset_is_census_data  
    def get_feature_dataset_presence_matrix(self, upper=None, lower=None, measurement_name=None, fmt=None, todense=None):
        if upper is None or lower is None:
            raise ValueError("No upper and/or lower bound for slicing was provided. Dataset is too large to fit in memory. \
                Memory-Efficient queries are not supported yet.")
        elif measurement_name is None:
            raise ValueError("measurement_name was not provided")
        elif fmt is not None and fmt not in ["scipy", "pyarrow"]:
            raise ValueError("feature dataset presence matrix only supports 'scipy' or 'pyarrow' formats")
        todense = todense if todense is not None else False
        fmt = fmt if fmt is not None else "scipy"
        if todense and fmt!="scipy":
            raise ValueError("dense representation only available in scipy format")
        with cellxgene_census.open_soma(
            census_version=self.census_version
        ) as census:
            n_obs = len(census[self.dataset][self.organism].obs)
            n_var = len(census[self.dataset][self.organism].ms[measurement_name].var) 
            fMatrix = census[self.dataset][self.organism].ms[measurement_name]["feature_dataset_presence_matrix"]
            slc = fMatrix.read((slice(0, 5),)).coos((n_obs,n_var))
            out = self.fmt_cellxgene_data(slc, fmt)
            return out if not todense else out.todense()


if __name__ == "__main__":
    # TODO: tmp, run testing suite when this file is called as main
    print("running tests for census resource")
    print("instantiating resource")
    resource = CensusResource()
    cell_value_filter = "tissue == 'brain' and sex == 'male'"
    cell_column_names = ["assay", "cell_type", "tissue"]
    gene_value_filter = "feature_id in ['ENSG00000161798', 'ENSG00000188229']"
    gene_column_names = ["feature_name", "feature_reference"]
    print("getting cell metadata as pandas dataframe")
    obsdf = resource.get_cell_metadata(value_filter=cell_value_filter, column_names=cell_column_names, fmt="pandas")
    print("success!")
    print(obsdf.head())
    print("geting gene metadata as pyarrow")
    varpyarrow = resource.get_gene_metadata(value_filter=gene_value_filter, column_names=gene_column_names, fmt="pyarrow", measurement_name="RNA")
    print("success!")
    print(varpyarrow)
    print("getting sample count matrix, checking todense() and scipy")
    Xslice = resource.get_measurement_matrix(upper=5, lower=0, measurement_name="RNA", fmt="scipy", todense=True)
    print("success")
    print(Xslice)
    print("getting feature presence matrix, checking pyarrow")
    FMslice = resource.get_feature_dataset_presence_matrix(upper=5, lower=0, measurement_name="RNA", fmt="pyarrow", todense=False)
    print("success")
    print(FMslice)
    print("all tests passed")