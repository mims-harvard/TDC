import numpy as np
import scipy.sparse as sp

from ...utils.load import pd_load, download_wrapper


class GeneformerTokenizer:
    """
    Uses Geneformer Utils to parse zero-shot model server requests for tokenizing single-cell gene expression data.

    Tokenizer source code: https://github.com/amva13/geneformer/blob/main/geneformer/tokenizer.py
    """

    def __init__(
        self,
        path=None,
        custom_attr_name_dict=None,
        nproc=1,
        max_input_size=4096,
    ):
        path = path or "./data"
        download_wrapper("geneformer_gene_median_dictionary", path,
                         ["geneformer_gene_median_dictionary"])
        download_wrapper("geneformer_gene_name_id_dict", path,
                         ["geneformer_gene_name_id_dict"])
        download_wrapper("geneformer_token_dictionary", path,
                         ["geneformer_token_dictionary"])
        self.gene_median_dict = pd_load("geneformer_gene_median_dictionary",
                                        path=path)
        self.gene_name_id_dict = pd_load("geneformer_gene_name_id_dict",
                                         path=path)
        self.gene_token_dict = pd_load("geneformer_token_dictionary", path=path)
        self.custom_attr_name_dict = custom_attr_name_dict
        self.nproc = nproc

        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        self.genelist_dict = dict(
            zip(self.gene_keys, [True] * len(self.gene_keys)))

        self.max_input_size = max_input_size

    @classmethod
    def rank_genes(cls, gene_vector, gene_tokens):
        """
        Rank gene expression vector.
        """
        # sort by median-scaled gene values
        sorted_indices = np.argsort(-gene_vector)
        return gene_tokens[sorted_indices]

    def tokenize_cell_vectors(self,
                              cell_vector_adata,
                              target_sum=10_000,
                              chunk_size=512,
                              ensembl_id="ensembl_id",
                              ncounts="ncounts"):
        """
        Tokenizing single-cell gene expression vectors formatted as anndata types.

        """
        adata = cell_vector_adata
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        coding_miRNA_loc = np.where([
            self.genelist_dict.get(i, False) for i in adata.var[ensembl_id]
        ])[0]
        norm_factor_vector = np.array([
            self.gene_median_dict[i]
            for i in adata.var[ensembl_id][coding_miRNA_loc]
        ])
        coding_miRNA_ids = adata.var[ensembl_id][coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict.get(i, 0) for i in coding_miRNA_ids])

        try:
            _ = adata.obs["filter_pass"]
        except KeyError:
            var_exists = False
        else:
            var_exists = True

        if var_exists:
            filter_pass_loc = np.where(
                [i == 1 for i in adata.obs["filter_pass"]])[0]
        elif not var_exists:
            print(
                f"The anndata object has no column attribute 'filter_pass'; tokenizing all cells."
            )
            filter_pass_loc = np.array([i for i in range(adata.shape[0])])

        tokenized_cells = []

        for i in range(0, len(filter_pass_loc), chunk_size):
            idx = filter_pass_loc[i:i + chunk_size]

            n_counts = adata[idx].obs[ncounts].values[:, None]
            X_view = adata[idx, coding_miRNA_loc].X
            X_norm = (X_view / n_counts * target_sum / norm_factor_vector)
            X_norm = sp.csr_matrix(X_norm)

            tokenized_cells.append([
                self.rank_genes(X_norm[i].data, coding_miRNA_tokens[
                    X_norm[i].indices])[:self.max_input_size]
                for i in range(X_norm.shape[0])
            ])

            # add custom attributes for subview to dict
            if self.custom_attr_name_dict is not None:
                for k in file_cell_metadata.keys():
                    file_cell_metadata[k] += adata[idx].obs[k].tolist()
            else:
                file_cell_metadata = None

        return tokenized_cells, file_cell_metadata
