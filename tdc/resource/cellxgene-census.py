# TODO: tmp fix
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"
# TODO: find better fix or encode in environment / docker ^^^
import cellxgene_census
import gget
import tiledbsoma


class CensusResource:

    _CENSUS_DATA = "census_data"
    _CENSUS_META = "census_info"
    _FEATURE_PRESENCE = "feature_dataset_presence_matrix"
    _LATEST_CENSUS = "2023-12-15"
    _HUMAN = "homo_sapiens"

    class decorators:

        @classmethod
        def check_dataset_is_census_data(cls, func):
            # @wraps(func)
            def check(*args, **kwargs):
                self = args[0]
                self.dataset = self._CENSUS_DATA
                return func(*args, **kwargs)

            return check

        @classmethod
        def check_dataset_is_census_info(cls, func):

            def check(*args, **kwargs):
                self = args[0]
                self.dataset = self._CENSUS_META
                return func(*args, **kwargs)

            return check

    def __init__(self, census_version=None, organism=None):
        """Initialize the Census Resource.
        
        Args:
            census_version (str): The date of the census data release in YYYY- 
            TODO: complete
        """
        self.census_version = census_version if census_version is not None else self._LATEST_CENSUS
        self.organism = organism if organism is not None else self._HUMAN
        self.dataset = None  # variable to set target census collection to either info or data

    def fmt_cellxgene_data(self, tiledb_ptr, fmt=None):
        if fmt is None:
            raise Exception(
                "format not provided to fmt_cellxgene_data(), please provide fmt variable"
            )
        elif fmt == "pandas":
            return tiledb_ptr.concat().to_pandas()
        elif fmt == "pyarrow":
            return tiledb_ptr.concat()
        elif fmt == "scipy":
            return tiledb_ptr.concat().to_scipy()
        else:
            raise Exception(
                "fmt not in [pandas, pyarrow, scipy] for fmt_cellxgene_data()")

    @decorators.check_dataset_is_census_data
    def get_cell_metadata(self, value_filter=None, column_names=None, fmt=None):
        """Get the cell metadata (obs) data from the Census API"""
        if value_filter is None:
            raise Exception(
                "No value filter was provided, dataset is too large to fit in memory. \
                            Memory-Efficient queries are not supported yet.")
        fmt = fmt if fmt is not None else "pandas"
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            obs = census[self.dataset][self.organism].obs
            obsread = None
            if column_names:
                obsread = obs.read(value_filter=value_filter,
                                   column_names=column_names)
            else:
                obsread = obs.read(value_filter=value_filter)
            return self.fmt_cellxgene_data(obsread, fmt)

    @decorators.check_dataset_is_census_data
    def get_gene_metadata(self,
                          value_filter=None,
                          column_names=None,
                          measurement_name=None,
                          fmt=None):
        """Get the gene metadata (var) data from the Census API"""
        if value_filter is None:
            raise Exception(
                "No value filter was provided, dataset is too large to fit in memory. \
                Memory-Efficient queries are not supported yet.")
        elif measurement_name is None:
            raise ValueError("measurment_name must be provided , i.e. 'RNA'")
        fmt = fmt if fmt is not None else "pandas"
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            var = census[self.dataset][self.organism].ms[measurement_name].var
            varread = None
            if column_names:
                varread = var.read(value_filter=value_filter,
                                   column_names=column_names)
            else:
                varread = var.read(value_filter=value_filter)
            return self.fmt_cellxgene_data(varread, fmt)

    @decorators.check_dataset_is_census_data
    def get_measurement_matrix(self,
                               upper=None,
                               lower=None,
                               value_adjustment=None,
                               measurement_name=None,
                               fmt=None,
                               todense=None):
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
            raise Exception(
                "No upper and/or lower bound for slicing was provided. Dataset is too large to fit in memory. \
                Memory-Efficient queries are not supported yet.")
        elif measurement_name is None:
            raise Exception("measurement_name was not provided.")
        elif fmt is not None and fmt not in ["scipy", "pyarrow"]:
            raise ValueError(
                "measurement_matrix only supports 'scipy' or 'pyarrow' format")
        value_adjustment = value_adjustment if value_adjustment is not None else "raw"
        todense = todense if todense is not None else False
        fmt = fmt if fmt is not None else "scipy"
        if todense and fmt != "scipy":
            raise ValueError(
                "dense representation only available in scipy format")
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            n_obs = len(census[self.dataset][self.organism].obs)
            n_var = len(
                census[self.dataset][self.organism].ms[measurement_name].var)
            X = census[self.dataset][
                self.organism].ms[measurement_name].X[value_adjustment]
            slc = X.read((slice(lower, upper),)).coos((n_obs, n_var))
            out = self.fmt_cellxgene_data(slc, fmt)
            return out if not todense else out.todense()

    @decorators.check_dataset_is_census_data
    def get_feature_dataset_presence_matrix(self,
                                            upper=None,
                                            lower=None,
                                            measurement_name=None,
                                            fmt=None,
                                            todense=None):
        if upper is None or lower is None:
            raise ValueError(
                "No upper and/or lower bound for slicing was provided. Dataset is too large to fit in memory. \
                Memory-Efficient queries are not supported yet.")
        elif measurement_name is None:
            raise ValueError("measurement_name was not provided")
        elif fmt is not None and fmt not in ["scipy", "pyarrow"]:
            raise ValueError(
                "feature dataset presence matrix only supports 'scipy' or 'pyarrow' formats"
            )
        todense = todense if todense is not None else False
        fmt = fmt if fmt is not None else "scipy"
        if todense and fmt != "scipy":
            raise ValueError(
                "dense representation only available in scipy format")
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            n_obs = len(census[self.dataset][self.organism].obs)
            n_var = len(
                census[self.dataset][self.organism].ms[measurement_name].var)
            fMatrix = census[self.dataset][self.organism].ms[measurement_name][
                "feature_dataset_presence_matrix"]
            slc = fMatrix.read((slice(0, 5),)).coos((n_obs, n_var))
            out = self.fmt_cellxgene_data(slc, fmt)
            return out if not todense else out.todense()

    @decorators.check_dataset_is_census_info
    def get_metadata(self):
        """Get the metadata for the Cell Census."""
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            return census[self.dataset]["summary"]

    @decorators.check_dataset_is_census_info
    def get_dataset_metadata(self):
        """Get the metadata for the Cell Census's datasets."""
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            return census[self.dataset]["datasets"]

    @decorators.check_dataset_is_census_info
    def get_cell_count_metadata(self):
        """Get the cell counts across cell metadata for the Cell Census."""
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            return census[self.dataset]["summary_cell_counts"]

    @decorators.check_dataset_is_census_data
    def query_measurement_matrix(self,
                                 value_filter=None,
                                 value_adjustment=None,
                                 measurement_name=None,
                                 fmt=None,
                                 todense=None):
        """Query the Census Measurement Matrix. Function returns a Python generator.

        Args:
            value_filter (_type_, optional): _description_. Defaults to None.
            value_adjustment (_type_, optional): _description_. Defaults to None.
            measurement_name (_type_, optional): _description_. Defaults to None.
            fmt (_type_, optional): _description_. Defaults to None.
            todense (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            Exception: _description_
            ValueError: _description_
            ValueError: _description_
            
        Yields:
            a slice of the output query in the specified format
        """
        if value_filter is None:
            raise ValueError(
                "query_measurement_matrix expects a value_filter. if you don't plan to apply a filter, use get_measurement_matrix()"
            )
        elif measurement_name is None:
            raise Exception("measurement_name was not provided.")
        elif fmt is not None and fmt not in ["scipy", "pyarrow"]:
            raise ValueError(
                "measurement_matrix only supports 'scipy' or 'pyarrow' format")
        value_adjustment = value_adjustment if value_adjustment is not None else "raw"
        todense = todense if todense is not None else False
        fmt = fmt if fmt is not None else "scipy"
        if todense and fmt != "scipy":
            raise ValueError(
                "dense representation only available in scipy format")
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            organism = census[self.dataset][self.organism]
            query = organism.axis_query(
                measurement_name=measurement_name,
                obs_query=tiledbsoma.AxisQuery(value_filter=value_filter))
            it = query.X(value_adjustment).tables()
            for slc in it:
                out = self.fmt_cellxgene_data(slc, fmt)
                out = out if not todense else out.todense()
                yield out

    @classmethod
    def gget_czi_cellxgene(cls, **kwargs):
        """Wrapper for cellxgene gget()
            https://chanzuckerberg.github.io/cellxgene-census/notebooks/api_demo/census_gget_demo.html
            Support for AnnData or DataFrame. Params included below
            
            General args:
            - species        Choice of 'homo_sapiens' or 'mus_musculus'. Default: 'homo_sapiens'.
            - gene           Str or list of gene name(s) or Ensembl ID(s), e.g. ['ACE2', 'SLC5A1'] or ['ENSG00000130234', 'ENSG00000100170']. Default: None.
                            NOTE: Set ensembl=True when providing Ensembl ID(s) instead of gene name(s).
                            See https://cellxgene.cziscience.com/gene-expression for examples of available genes.
            - ensembl        True/False (default: False). Set to True when genes are provided as Ensembl IDs.
            - column_names   List of metadata columns to return (stored in AnnData.obs when meta_only=False).
                            Default: ["dataset_id", "assay", "suspension_type", "sex", "tissue_general", "tissue", "cell_type"]
                            For more options see: https://api.cellxgene.cziscience.com/curation/ui/#/ -> Schemas -> dataset
            - meta_only      True/False (default: False). If True, returns only metadata dataframe (corresponds to AnnData.obs).
            - census_version Str defining version of Census, e.g. "2023-05-15" or "latest" or "stable". Default: "stable".
            - verbose        True/False whether to print progress information. Default True.
            - out            If provided, saves the generated AnnData h5ad (or csv when meta_only=True) file with the specified path. Default: None.

        Cell metadata attributes:
            - tissue                          Str or list of tissue(s), e.g. ['lung', 'blood']. Default: None.
                                            See https://cellxgene.cziscience.com/gene-expression for examples of available tissues.
            - cell_type                       Str or list of celltype(s), e.g. ['mucus secreting cell', 'neuroendocrine cell']. Default: None.
                                            See https://cellxgene.cziscience.com/gene-expression and select a tissue to see examples of available celltypes.
            - development_stage               Str or list of development stage(s). Default: None.
            - disease                         Str or list of disease(s). Default: None.
            - sex                             Str or list of sex(es), e.g. 'female'. Default: None.
            - is_primary_data                 True/False (default: True). If True, returns only the canonical instance of the cellular observation.
                                            This is commonly set to False for meta-analyses reusing data or for secondary views of data.
            - dataset_id                      Str or list of CELLxGENE dataset ID(s). Default: None.
            - tissue_general_ontology_term_id Str or list of high-level tissue UBERON ID(s). Default: None.
                                            Also see: https://github.com/chanzuckerberg/single-cell-data-portal/blob/9b94ccb0a2e0a8f6182b213aa4852c491f6f6aff/backend/wmg/data/tissue_mapper.py
            - tissue_general                  Str or list of high-level tissue label(s). Default: None.
                                            Also see: https://github.com/chanzuckerberg/single-cell-data-portal/blob/9b94ccb0a2e0a8f6182b213aa4852c491f6f6aff/backend/wmg/data/tissue_mapper.py
            - tissue_ontology_term_id         Str or list of tissue ontology term ID(s) as defined in the CELLxGENE dataset schema. Default: None.
            - assay_ontology_term_id          Str or list of assay ontology term ID(s) as defined in the CELLxGENE dataset schema. Default: None.
            - assay                           Str or list of assay(s) as defined in the CELLxGENE dataset schema. Default: None.
            - cell_type_ontology_term_id      Str or list of celltype ontology term ID(s) as defined in the CELLxGENE dataset schema. Default: None.
            - development_stage_ontology_term_id        Str or list of development stage ontology term ID(s) as defined in the CELLxGENE dataset schema. Default: None.
            - disease_ontology_term_id        Str or list of disease ontology term ID(s) as defined in the CELLxGENE dataset schema. Default: None.
            - donor_id                        Str or list of donor ID(s) as defined in the CELLxGENE dataset schema. Default: None.
            - self_reported_ethnicity_ontology_term_id  Str or list of self reported ethnicity ontology ID(s) as defined in the CELLxGENE dataset schema. Default: None.
            - self_reported_ethnicity         Str or list of self reported ethnicity as defined in the CELLxGENE dataset schema. Default: None.
            - sex_ontology_term_id            Str or list of sex ontology ID(s) as defined in the CELLxGENE dataset schema. Default: None.
            - suspension_type                 Str or list of suspension type(s) as defined in the CELLxGENE dataset schema. Default: None.

        Returns AnnData object (when meta_only=False) or dataframe (when meta_only=True).

        """
        gget.setup("cellxgene")
        return gget.cellxgene(**kwargs)


if __name__ == "__main__":
    # TODO: tmp, run testing suite when this file is called as main
    print("running tests for census resource")
    print("instantiating resource")
    resource = CensusResource()
    cell_value_filter = "tissue == 'brain' and sex == 'male'"
    cell_column_names = ["assay", "cell_type", "tissue"]
    gene_value_filter = "feature_id in ['ENSG00000161798', 'ENSG00000188229']"
    gene_column_names = ["feature_name", "feature_length"]
    print("getting cell metadata as pandas dataframe")
    obsdf = resource.get_cell_metadata(value_filter=cell_value_filter,
                                       column_names=cell_column_names,
                                       fmt="pandas")
    print("success!")
    print(obsdf.head())
    print("geting gene metadata as pyarrow")
    varpyarrow = resource.get_gene_metadata(value_filter=gene_value_filter,
                                            column_names=gene_column_names,
                                            fmt="pyarrow",
                                            measurement_name="RNA")
    print("success!")
    print(varpyarrow)
    print("getting sample count matrix, checking todense() and scipy")
    Xslice = resource.get_measurement_matrix(upper=5,
                                             lower=0,
                                             measurement_name="RNA",
                                             fmt="scipy",
                                             todense=True)
    print("success")
    print(Xslice)
    print("getting feature presence matrix, checking pyarrow")
    FMslice = resource.get_feature_dataset_presence_matrix(
        upper=5, lower=0, measurement_name="RNA", fmt="pyarrow", todense=False)
    print("success")
    print(FMslice)
    print("all tests passed")
