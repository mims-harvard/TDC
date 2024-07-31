from ..utils import general_load
from ..utils.load import download_wrapper, load_json_from_txt_file

import pandas as pd
import os
import torch


class PINNACLE:
    """
    PINNACLE is a class for loading and manipulating the PINNACLE networks and embeddings.
    @article{
        Li2023,
        author = "Michelle Li",
        title = "{PINNACLE}",
        year = "2023",
        month = "4",
        url = "https://figshare.com/articles/software/AWARE/22708126",
        doi = "10.6084/m9.figshare.22708126.v5"
    }
    """

    def __init__(self, path="./data"):
        self.ppi_name = "pinnacle_global_ppi_edgelist"
        self.cell_tissue_mg_name = "cell_tissue_mg_edgelist"
        self.ppi = general_load(self.ppi_name, path, " ")
        self.ppi.columns = ["Protein A", "Protein B"]
        self.cell_tissue_mg = general_load(
            self.cell_tissue_mg_name, path,
            "\t")  # use tab as names were left with spaces
        self.cell_tissue_mg.columns = ["Tissue", "Cell"]
        self.embeds_name = "pinnacle_protein_embed"
        # self.embeds = resource_dataset_load(self.embeds_name, path, [self.embeds_name])
        self.embeds_name = download_wrapper(self.embeds_name, path,
                                            self.embeds_name)
        self.embeds = torch.load(os.path.join(path, self.embeds_name + ".pth"))
        self.keys = load_json_from_txt_file("pinnacle_labels_dict", path)

    def get_ppi(self):
        return self.ppi

    def get_mg(self):
        return self.cell_tissue_mg

    def get_embeds_raw(self):
        return self.embeds

    def get_keys(self):
        protein_names_celltypes = [
            p for p in zip(self.keys["Cell Type"], self.keys["Name"])
            if not (p[0].startswith("BTO") or p[0].startswith("CCI") or
                    p[0].startswith("Sanity"))
        ]
        proteins = pd.DataFrame.from_dict({
            "target": [n for _, n in protein_names_celltypes],
            "cell type": [c for c, _ in protein_names_celltypes]
        })
        proteins.drop_duplicates()
        return proteins

    def get_embeds(self):
        prots = self.get_keys()
        emb = self.get_embeds_raw()
        # nemb = {'--'.join(prots.iloc[k]): v for k, v in emb.items()}
        x = {}
        ctr = 0
        for _, v in emb.items():
            if isinstance(v, torch.Tensor):
                if v.size()[0] == 1:
                    k = "--".join(prots.iloc[ctr])
                    ctr += 1
                    x[k] = v.detach().numpy()
                    continue
                for t in v:
                    assert len(t.size()) == 1, t.size()
                    k = "--".join(prots.iloc[ctr])
                    ctr += 1
                    x[k] = t.detach().numpy()
            else:
                raise Exception("encountered non-tensor")
        assert len(x) == len(prots), "dict len {} vs keys length {}".format(
            len(x), len(prots))
        df = pd.DataFrame.from_dict(x)
        df = df.transpose()
        assert len(df) == len(
            x), "dims not mantained when translated to pandas. {} vs {}".format(
                len(df), len(x))
        return df
