import torch.nn as nn


class scVI(nn.Module):

    def __init__(self):
        import scvi as scvi_package

        super().__init__()
        self.model = None
        self.var_names = None

    def forward(self, adata):
        try:
            self.prepare_data(adata)

        except Exception as error:
            print("No var names found in SCVI reference vars")
            print(f"adata.var.index must include some of {self.var_names}")

            # See docstring for more information. Expect input data to have index
            # ordered and dimensions per scVI specifications
            self.force_data_match(adata)

        # getting variational autoencoder
        vae_q = self.model.load_query_data(adata, self.model)
        vae_q.is_trained = True

        return vae_q.get_latent_representation()

    def load(self):
        import scvi as scvi_package
        import os

        from tdc.multi_pred.anndata_dataset import DataLoader
        from model_server.model_loaders import scvi_loader

        if not os.path.isdir("scvi_model"):
            loader = scvi_loader.scVILoader()
            loader.load("2024-07-01")

        # adata object needed for loading model
        adata = DataLoader("scvi_test_dataset",
                           "./data",
                           dataset_names=["scvi_test_dataset"],
                           no_convert=True).adata

        # See docstring for more information. Expect index
        # ordered and dimensions per scVI specifications
        self.force_data_match(adata)

        #instantiate SCVI model (not callable, just used to get VAE)
        self.model = scvi_package.model.SCVI.load('scvi_model', adata)
        self.var_names = adata.var.index

        print("loaded scVI:")
        print(f"{self.model}")

        return self.model

    def prepare_data(self, adata):
        import numpy as np

        assert True in np.isin(adata.var.index, self.var_names)

        adata.obs["batch"] = "unassigned"
        self.model.prepare_query_anndata(adata, self.model)

    def force_data_match(self, adata):
        '''
        Input data is expected to have index ordered and dimensions as per scVI specifications.
        For more information visit:
        https://huggingface.co/datasets/scvi-tools/DATASET-FOR-UNIT-TESTING-1/tree/main
        '''
        import torch
        import numpy as np

        metadata = torch.load("scvi_model/model.pt",
                              map_location=torch.device('cpu'))

        # setting indices that match
        adata.var.index = metadata["attr_dict"]["registry_"][
            "field_registries"]["X"]["state_registry"]["column_names"]

        # Padding X so dimensions match. Need 8000 columns because scVI was trained using adata
        # containing 8000 genes. This is the number of var indices extracted from metadata above.
        additional_columns = np.zeros(
            (adata.X.shape[0], 8000 - adata.X.shape[1]))
        adata.X = np.hstack([adata.X, additional_columns])

        # getting a batch name that matches
        adata.obs['batch'] = metadata["attr_dict"]["registry_"][
            "field_registries"]["batch"]["state_registry"][
                "categorical_mapping"][0]
