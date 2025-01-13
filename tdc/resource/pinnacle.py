from ..utils import general_load
from ..utils.load import download_wrapper, load_json_from_txt_file, zip_data_download_wrapper

import pandas as pd
import os
import torch


class PINNACLE:
    """
    PINNACLE is a class for loading and manipulating the PINNACLE networks and embeddings.
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

    def get_exp_data(self, seed=1, split="train"):
        if split not in ["train", "val", "test"]:
            raise ValueError("{} not a valid split".format(split))
        if seed < 1 or seed > 10:
            raise ValueError(f'{seed} is not a valid seed in range 1-10.')
        filename = "pinnacle_output{}".format(seed)
        # clean data directory
        file_list = os.listdir("./data")
        for file in file_list:
            try:
                os.remove(os.path.join("./data", file))
            except:
                continue
        print("downloading pinancle zip data...")
        zip_data_download_wrapper(
            filename, "./data",
            ["pinnacle_output{}".format(x) for x in range(1, 11)])
        print("success!")
        # Get non-csv files and remove them
        non_csv_files = [
            f for f in os.listdir("./data") if not f.endswith(".csv")
        ]
        for x in non_csv_files:
            try:
                os.remove("./data/{}".format(x))
            except:
                continue
        # Get a list of all CSV files in the unzipped folder
        csv_files = [f for f in os.listdir("./data") if f.endswith(".csv")]
        if not csv_files:
            raise Exception("no csv")
        x = []
        print("iterating over csv files...")
        for file in csv_files:
            print("got file {}".format(file))
            if "_{}_".format(split) not in file:
                os.remove("./data/{}".format(file))
                continue
            print("reading into pandas...")
            df = pd.read_csv("./data/{}".format(file))
            cell = file.split("_")[-1]
            cell = cell.split(".")[0]
            df["cell_type_label"] = cell
            disease = "IBD" if "3767" in file else "RA"
            df["disease"] = disease
            x.append(df)
            os.remove("./data/{}".format(file))
        return pd.concat(x, axis=0, ignore_index=True)
