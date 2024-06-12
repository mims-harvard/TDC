import cellxgene_census
import gget
import pandas as pd
from scipy.sparse import csr_matrix
import tiledbsoma


class CensusResource:

    _CENSUS_DATA = "census_data"
    _CENSUS_META = "census_info"
    _FEATURE_PRESENCE = "feature_dataset_presence_matrix"
    _LATEST_CENSUS = "stable"
    _HUMAN = "homo_sapiens"

    class decorators:

        @classmethod
        def check_dataset_is_census_data(cls, func):
            """Sets self.dataset to census_data"""

            def check(*args, **kwargs):
                self = args[0]
                self.dataset = self._CENSUS_DATA
                return func(*args, **kwargs)

            return check

        @classmethod
        def check_dataset_is_census_info(cls, func):
            """Sets self.dataset to census_data"""

            def check(*args, **kwargs):
                self = args[0]
                self.dataset = self._CENSUS_META
                return func(*args, **kwargs)

            return check

        @classmethod
        def slice_checks_X_and_FM(cls, func):
            """Decorator for:
            1. functions that need X and feature presence matrix apply slicing if not filtering
            2. functions with a todense() option abide by required formatting
            3. functions requiring a measurement name provide a measurement name
            4. fmt is a valid format
            asserts these requirements hold in input arguments."""

            def check(*args, **kwargs):
                if "upper" in kwargs:
                    upper, lower = kwargs.get('upper',
                                              None), kwargs.get("lower", None)
                    if upper is None or lower is None:
                        raise Exception(
                            "No upper and/or lower bound for slicing was provided. Dataset is too large to fit in memory. \
                            Memory-Efficient queries are not supported yet.")
                fmt = kwargs.get("fmt")
                fmt = fmt if fmt is not None else "pandas"
                if "todense" in kwargs:
                    todense = kwargs.get("todense")
                    kwargs[
                        "todense"] = todense if todense is not None else False
                    if todense and fmt != "scipy":
                        raise ValueError(
                            "dense representation only available in scipy format"
                        )
                measurement_name = kwargs.get("measurement_name")
                if measurement_name is None:
                    raise Exception("measurement_name was not provided.")
                kwargs["fmt"] = fmt if fmt is not None else "pandas"
                return func(*args, **kwargs)

            return check

    def __init__(self, census_version=None, organism=None):
        """Initialize the Census Resource.
        
        Args:
            census_version (str): The date of the census data release to use 
            organism (str): string for census data organism to query data for. defaults to human.
        """
        self.census_version = census_version if census_version is not None else self._LATEST_CENSUS
        self.organism = organism if organism is not None else self._HUMAN
        self.dataset = None  # variable to set target census collection to either info or data

    def fmt_cellxgene_data(self,
                           tiledb_ptr,
                           fmt=None,
                           input_is_table=False,
                           is_csr=False,
                           is_soma=False):
        """Transform TileDB DataFrame or SparseNDArray to one of the supported API formats.

        Args:
            tiledb_ptr (TileDB DataFrame or SparseNDArray): pointer to the TileDB DataFrame
            fmt (str, optional): deisgnates a format to transfowm TileDB data to. Defaults to None.

        Raises:
            Exception: if no format is provided
            Exception: if format is not a valid option

        Returns:
            The dataset in selected format if it's a valid format
        """
        if fmt is None:
            raise Exception(
                "format not provided to fmt_cellxgene_data(), please provide fmt variable"
            )
        elif is_soma:
            import tiledb
            uri = tiledb_ptr.uri
            config = tiledb.Config({
                'vfs.s3.region': 'us-west-2',
            })
            ctx = tiledb.Ctx(config=config)
            with tiledb.open(uri, ctx=ctx, mode="r") as df:
                return pd.DataFrame(df[:])
        elif fmt == "pandas" and is_csr:
            return pd.DataFrame(tiledb_ptr.toarray())
        elif fmt == "pandas" and not input_is_table:
            return tiledb_ptr.concat().to_pandas()
        elif fmt == "pandas":
            return tiledb_ptr.to_pandas()
        elif fmt == "pyarrow" and not input_is_table:
            return tiledb_ptr.concat()
        elif fmt == "pyarrow":
            return tiledb_ptr
        elif fmt == "scipy" and not input_is_table:
            return tiledb_ptr.concat().to_scipy()
        elif fmt == "scipy":
            return csr_matrix(tiledb_ptr.to_pandas().values)
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

    @decorators.slice_checks_X_and_FM
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
            upper (int, optional): upper bound on the slice to obtain. Defaults to None.
            lower (int, optional): lower bound on the slice to obtain. Defaults to None.
            value_adjustment (str, optional): designates the type of count desired for this measurement. Defaults to None.
            measurement_name (str, optional): name of measurement, i.e. 'raw'. Defaults to None.

        Returns:
            A slice from the count matrix in the specified format. If `todense` is True, then a dense scipy array will be returned.
        """
        value_adjustment = value_adjustment if value_adjustment is not None else "raw"
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            n_obs = len(census[self.dataset][self.organism].obs)
            n_var = len(
                census[self.dataset][self.organism].ms[measurement_name].var)
            X = census[self.dataset][
                self.organism].ms[measurement_name].X[value_adjustment]
            slc = X.read([slice(lower, upper)]).coos((n_obs, n_var))
            out = self.fmt_cellxgene_data(slc, fmt)
            return out if not todense else out.todense()

    @decorators.slice_checks_X_and_FM
    @decorators.check_dataset_is_census_data
    def get_feature_dataset_presence_matrix(self,
                                            upper=None,
                                            lower=None,
                                            measurement_name=None,
                                            fmt=None,
                                            todense=None):
        """Gets a slice from the feature_dataset_presence_matrix for a given measurement_name

        Args:
            upper (int, optional): upper bound on the slice. Defaults to None.
            lower (int, optional): lower bound on the slice. Defaults to None.
            measurement_name (str, optional): measurment_name for the query i.e. 'rna'. Defaults to None.
            fmt (str, optional): output format desired for the output dataset. Defaults to None.
            todense (bool, optional): if True, returns scipy dense representation. Defaults to None.

        Returns:
            dataset in desired format
        """
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            n_obs = len(census[self.dataset][self.organism].obs)
            n_var = len(
                census[self.dataset][self.organism].ms[measurement_name].var)
            fMatrix = census[self.dataset][self.organism].ms[measurement_name][
                "feature_dataset_presence_matrix"]
            slc = fMatrix.read((slice(lower, upper),)).coos((n_obs, n_var))
            out = self.fmt_cellxgene_data(slc, fmt)
            return out if not todense else out.todense()

    @decorators.check_dataset_is_census_data
    def get_feature_dataset_presence_matrix_entry(self,
                                                  dataset_name,
                                                  measurement_name=None,
                                                  fmt=None,
                                                  todense=None):
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            fMatrix = cellxgene_census.get_presence_matrix(
                census,
                organism=self.organism,
                measurement_name=measurement_name)
            meta_df = self.get_dataset_metadata()
            dataset_id = meta_df[meta_df["dataset_title"] == dataset_name].index
            entry = fMatrix[dataset_id]
            out = self.fmt_cellxgene_data(entry, "pandas", is_csr=True)
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
            return self.fmt_cellxgene_data(census[self.dataset]["datasets"],
                                           fmt="pandas",
                                           is_soma=True)

    @decorators.check_dataset_is_census_info
    def get_cell_count_metadata(self):
        """Get the cell counts across cell metadata for the Cell Census."""
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            return census[self.dataset]["summary_cell_counts"]

    @decorators.slice_checks_X_and_FM
    @decorators.check_dataset_is_census_data
    def query_measurement_matrix(self,
                                 value_filter=None,
                                 value_adjustment=None,
                                 measurement_name=None,
                                 fmt=None,
                                 todense=None,
                                 gene_filter=None):
        """Query the Census Measurement Matrix. Function returns a Python generator.

        Args:
            value_filter (str, optional): a valuer filter (obs) to apply to the query. Defaults to None.
            value_adjustment (str, optional): the type of count to obtain from count matricx for this measurement. Defaults to None.
            measurement_name (str, optional): measurement name to query, i.e. "RNA". Defaults to None.
            fmt (str, optional): output format for the output dataset. Defaults to None.
            todense (bool, optional): if True, will output a dense scipy array as the representation. Defaults to None.

        Yields:
            a slice of the output query in the specified format
        """
        import numpy
        value_adjustment = value_adjustment if value_adjustment is not None else "raw"
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            organism = census[self.dataset][self.organism]
            query = organism.axis_query(
                measurement_name=measurement_name,
                obs_query=tiledbsoma.AxisQuery(value_filter=value_filter)
                if value_filter else tiledbsoma.AxisQuery(),
                var_query=tiledbsoma.AxisQuery(
                    coords=(numpy.array(gene_filter),))
                if gene_filter else tiledbsoma.AxisQuery(),
            )
            it = query.X(value_adjustment).tables()
            for slc in it:
                out = self.fmt_cellxgene_data(slc, fmt, input_is_table=True)
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

    def get_anndata(self, add_embeddings=False, **kwargs):
        """
        Get AnnData object.

        """
        embeddings = None
        if add_embeddings and "emb_names" not in kwargs:
            embeddings = ["scvi", "geneformer"]
        else:
            embeddings = kwargs.get("emb_names")
        with cellxgene_census.open_soma(
                census_version=self.census_version) as census:
            adata = cellxgene_census.get_anndata(
                census=census,
                organism="Homo sapiens",
                var_value_filter=kwargs.get("var_value_filter"),
                obs_value_filter=kwargs.get("obs_value_filter"),
                obs_embeddings=embeddings,
            )
            return adata


if __name__ == "__main__":
    pass
