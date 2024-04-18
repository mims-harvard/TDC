from .data_feature_generator import DataFeatureGenerator


class CellXGeneFeatureGenerator(DataFeatureGenerator):

    @classmethod
    def get_dense_soma_dataframe(cls, dataset):
        return dataset[dataset["soma_data"] != 0]

    @classmethod
    def format_cellxgene_dataframe(cls, dataset):
        dataset.columns = ["cell_idx", "gene_idx", "expression"]
        return dataset
