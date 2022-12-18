import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class FC2(nn.Module):
    def __init__(self, in_features, out_features):
        super(FC2, self).__init__()

        dropout = 0
        self.bn = nn.BatchNorm1d(in_features)
        self.fc1 = nn.Linear(in_features, int(in_features / 2))
        self.fc2 = nn.Linear(int(in_features / 2), out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.bn(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DrugCombDataset(Dataset):
    def __init__(self, df, drug_features, cell_features):
        self.df = df
        self.drug_features = drug_features
        self.cell_features = cell_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        d1 = self.df.iloc[idx, 0]
        d2 = self.df.iloc[idx, 1]
        cell = self.df.iloc[idx, 2]

        # external features
        d1_fp = np.array(self.drug_features.loc[d1, "fps"])
        d2_fp = np.array(self.drug_features.loc[d2, "fps"])
        c_gn = self.cell_features[self.cell_features["Cell_Line_ID"] == cell][
            "CellLine"
        ].values[0]
        out_metric = torch.tensor(self.df.loc[idx, "Y"], dtype=torch.float)

        sample = {
            "d1_fp": d1_fp,
            "d2_fp": d2_fp,
            "c_gn": c_gn,
            "out_metric": out_metric,
        }

        return sample


class DrugEncoder(nn.Module):
    def __init__(self, num_drug_fp=167, fp_embed_size=32, out_size=64):
        super(DrugEncoder, self).__init__()

        # fingerprint
        self.dense_fp = nn.Linear(num_drug_fp, fp_embed_size)

        # no transformer
        self.FC2 = FC2(fp_embed_size, out_size)

    def forward(self, fp):
        fp = F.relu(self.dense_fp(fp))
        x = self.FC2(fp)

        return x


class CellEncoder(nn.Module):
    def __init__(self, gene_embed_size=256, num_genes=0, out_size=64):
        super(CellEncoder, self).__init__()
        self.dense_gene = nn.Linear(num_genes, gene_embed_size)
        self.FC2 = FC2(gene_embed_size, out_size)

    def forward(self, gn):
        gn = F.relu(self.dense_gene(gn))
        x = self.FC2(gn)

        return x


class Comb(nn.Module):
    def __init__(self, out_size=64, num_genes=0):
        super(Comb, self).__init__()
        self.drugEncoder = DrugEncoder()
        self.cellEncoder = CellEncoder(num_genes=num_genes)
        self.fc_1 = FC2(out_size * 3, 64)
        self.fc_2 = nn.Linear(64, 1)

    def forward(self, d1, d2, c_):
        d1 = self.drugEncoder(d1)
        d2 = self.drugEncoder(d2)
        c_ = self.cellEncoder(c_)

        x = self.fc_1(torch.cat((d1, d2, c_), 1))
        out = self.fc_2(x)

        return out
