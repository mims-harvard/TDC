# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT

import warnings

warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import train_test_split
import sys

from ..utils import print_sys
from .single_cell import CellXGeneTemplate
from ..dataset_configs.config_map import scperturb_gene_datasets


def parse_single_pert(i):
    a = i.split('+')[0]
    b = i.split('+')[1]
    if a == 'ctrl':
        pert = b
    else:
        pert = a
    return pert


def parse_combo_pert(i):
    return i.split('+')[0], i.split('+')[1]


def parse_any_pert(p):
    if ('ctrl' in p) and (p != 'ctrl'):
        return [parse_single_pert(p)]
    elif 'ctrl' not in p:
        out = parse_combo_pert(p)
        return [out[0], out[1]]


class PerturbOutcome(CellXGeneTemplate):

    def __init__(self, name, path="./data", print_stats=False):
        super().__init__(name, path, print_stats)
        if name in scperturb_gene_datasets:
            self.is_gene = True
        else:
            self.is_gene = False

        if name == 'scperturb_gene_NormanWeissman2019':
            self.is_combo = True
        else:
            self.is_combo = False

    def get_mean_expression(self):
        raise ValueError("TODO")

    def get_DE_genes(self):
        raise ValueError("TODO")

    def get_dropout_genes(self):
        raise ValueError("TODO")

    def get_cellline_split(self,
                           ratios=[0.8, 0.1, 0.1],
                           random_state=42,
                           split_to_unseen=False,
                           remove_unseen=True):
        df = self.get_data()
        print("got data grouping by cell line")
        cell_line_groups = df.groupby("cell_line")
        print("groupby completed")
        cell_line_splits = {}
        for cell_line, cell_line_group in cell_line_groups:
            print("processing cell line", cell_line)
            control = cell_line_group[cell_line_group["perturbation"] ==
                                      "control"]
            cell_line_group = cell_line_group[cell_line_group["perturbation"] !=
                                              "control"]
            if not split_to_unseen:
                train, tmp = train_test_split(cell_line_group,
                                              test_size=ratios[1] + ratios[2],
                                              random_state=random_state)
                test, dev = train_test_split(tmp,
                                             test_size=ratios[2] /
                                             (ratios[1] + ratios[2]),
                                             random_state=random_state)
                filter_test = test["perturbation"].isin(train["perturbation"])
                filter_dev = dev["perturbation"].isin(train["perturbation"])
                adj = 0
                if remove_unseen:
                    lbef = len(test), len(dev)
                    test = test[~filter_test]
                    dev = dev[~filter_dev]
                    laft = len(test), len(dev)
                    adj = sum(lbef) - sum(laft)
                # TODO: filters might dilute test/dev siginificantly ...
                cell_line_splits[cell_line] = {
                    "control": control,
                    "train": train,
                    "test": test,
                    "dev": dev,
                    "adj": adj,
                }
            else:
                perturbs = cell_line_group["perturbation"].unique()
                perturbs_train, tmp = train_test_split(
                    perturbs,
                    test_size=ratios[1] + ratios[2],
                    random_state=random_state)
                perturbs_test, perturbs_dev = train_test_split(
                    tmp,
                    test_size=ratios[2] / (ratios[1] + ratios[2]),
                    random_state=random_state)
                cell_line_splits[cell_line] = {
                    "control":
                        control,
                    "train":
                        cell_line_group[
                            cell_line_group["perturbation"].isin(perturbs_train)
                        ],
                    "test":
                        cell_line_group[
                            cell_line_group["perturbation"].isin(perturbs_test)
                        ],
                    "dev":
                        cell_line_group[
                            cell_line_group["perturbation"].isin(perturbs_dev)]
                }
            print("done with cell line", cell_line)

        return cell_line_splits

    def get_perts_from_genes(self, genes, pert_list, type_='both'):
        """
            Returns all single/combo/both perturbations that include a gene
            """

        single_perts = [p for p in pert_list if ('ctrl' in p) and (p != 'ctrl')]
        combo_perts = [p for p in pert_list if 'ctrl' not in p]

        perts = []

        if type_ == 'single':
            pert_candidate_list = single_perts
        elif type_ == 'combo':
            pert_candidate_list = combo_perts
        elif type_ == 'both':
            pert_candidate_list = pert_list

        for p in pert_candidate_list:
            for g in genes:
                if g in parse_any_pert(p):
                    perts.append(p)
                    break
        return perts

    def get_genes_from_perts(self, perts):
        """
        Returns list of genes involved in a given perturbation list
        """

        if type(perts) is str:
            perts = [perts]
        gene_list = [p.split('+') for p in np.unique(perts)]
        gene_list = [item for sublist in gene_list for item in sublist]
        gene_list = [g for g in gene_list if g != 'ctrl']
        return np.unique(gene_list)

    def get_simulation_split_single(self,
                                    pert_list,
                                    train_gene_set_size=0.85,
                                    seed=1):
        unique_pert_genes = self.get_genes_from_perts(pert_list)

        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)

        ## a pre-specified list of genes
        train_gene_candidates = np.random.choice(
            unique_pert_genes,
            int(len(unique_pert_genes) * train_gene_set_size),
            replace=False)

        ## ood genes
        ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)

        pert_single_train = self.get_perts_from_genes(train_gene_candidates,
                                                      pert_list, 'single')
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list,
                                                  'single')

        #print(len(pert_single_train), len(unseen_single), len(pert_list))
        #assert len(unseen_single) + len(pert_single_train) == len(pert_list)

        return pert_single_train, unseen_single, {
            'unseen_single': unseen_single
        }

    def get_simulation_split(self,
                             pert_list,
                             train_gene_set_size=0.85,
                             combo_seen2_train_frac=0.85,
                             seed=1):

        unique_pert_genes = self.get_genes_from_perts(pert_list)

        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)

        ## a pre-specified list of genes
        train_gene_candidates = np.random.choice(
            unique_pert_genes,
            int(len(unique_pert_genes) * train_gene_set_size),
            replace=False)

        ## ood genes
        ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)

        pert_single_train = self.get_perts_from_genes(train_gene_candidates,
                                                      pert_list, 'single')
        pert_combo = self.get_perts_from_genes(train_gene_candidates, pert_list,
                                               'combo')
        pert_train.extend(pert_single_train)

        ## the combo set with one of them in OOD
        combo_seen1 = [
            x for x in pert_combo
            if len([t for t in x.split('+') if t in train_gene_candidates]) == 1
        ]
        pert_test.extend(combo_seen1)

        pert_combo = np.setdiff1d(pert_combo, combo_seen1)
        ## randomly sample the combo seen 2 as a test set, the rest in training set
        np.random.seed(seed=seed)
        pert_combo_train = np.random.choice(
            pert_combo,
            int(len(pert_combo) * combo_seen2_train_frac),
            replace=False)

        combo_seen2 = np.setdiff1d(pert_combo, pert_combo_train).tolist()
        pert_test.extend(combo_seen2)
        pert_train.extend(pert_combo_train)

        ## unseen single
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list,
                                                  'single')
        combo_ood = self.get_perts_from_genes(ood_genes, pert_list, 'combo')
        pert_test.extend(unseen_single)

        ## here only keeps the seen 0, since seen 1 is tackled above
        combo_seen0 = [
            x for x in combo_ood
            if len([t for t in x.split('+') if t in train_gene_candidates]) == 0
        ]
        pert_test.extend(combo_seen0)
        #assert len(combo_seen1) + len(combo_seen0) + len(unseen_single) + len(pert_train) + len(combo_seen2) == len(pert_list)

        return pert_train, pert_test, {
            'combo_seen0': combo_seen0,
            'combo_seen1': combo_seen1,
            'combo_seen2': combo_seen2,
            'unseen_single': unseen_single
        }

    def get_split(self,
                  ratios=[0.8, 0.1, 0.1],
                  unseen=False,
                  use_random=False,
                  random_state=42,
                  train_val_gene_set_size=0.75,
                  combo_seen2_train_frac=0.75,
                  remove_unseen=True):
        """obtain train/dev/test splits for each cell_line
        counterfactual prediction model is trained on a single cell line and then evaluated on same cell line
        and against new cell lines
        """

        if self.is_gene:
            ## use gene perturbation data split
            # check if this data has single or combo perturbations
            train_gene_set_size = train_val_gene_set_size
            combo_seen2_train_frac = combo_seen2_train_frac

            if self.is_combo:

                def map_name(x):
                    if x == 'control':
                        return 'ctrl'
                    else:
                        return '+'.join(
                            x.split('_')) if '_' in x else x + '+ctrl'

                self.adata.obs['condition'] = self.adata.obs.perturbation.apply(
                    lambda x: map_name(x))
                unique_perts = self.adata.obs.condition.unique()
                train, test, test_subgroup = self.get_simulation_split(
                    unique_perts, train_gene_set_size, combo_seen2_train_frac,
                    random_state)
                train, val, val_subgroup = self.get_simulation_split(
                    train, 0.9, 0.9, random_state)
            else:
                self.adata.obs['condition'] = self.adata.obs.perturbation.apply(
                    lambda x: x + '+ctrl' if x != 'control' else 'ctrl')
                unique_perts = self.adata.obs.condition.unique()

                train, test, test_subgroup = self.get_simulation_split_single(
                    unique_perts, train_gene_set_size, random_state)
                train, val, val_subgroup = self.get_simulation_split_single(
                    train, 0.9, random_state)

            map_dict = {x: 'train' for x in train}
            map_dict.update({x: 'val' for x in val})
            map_dict.update({x: 'test' for x in test})
            map_dict.update({'ctrl': 'train'})

            self.adata.obs['split'] = self.adata.obs['condition'].map(map_dict)
            adata_out = {
                "train": self.adata[self.adata.obs.split == 'train'],
                "dev": self.adata[self.adata.obs.split == 'val'],
                "test": self.adata[self.adata.obs.split == 'test']
            }
            subgroup = {
                'test_subgroup': test_subgroup,
                'dev_subgroup': val_subgroup
            }
            return adata_out, subgroup

        if not use_random:
            return self.get_cellline_split(split_to_unseen=unseen,
                                           ratios=ratios,
                                           random_state=random_state,
                                           remove_unseen=remove_unseen)
        df = self.get_data()
        # just do a random split, otherwise you'll split by cell line...
        control = df[df["perturbation"] == "control"]
        perturbs = df[df["perturbation"] != "control"]
        train, tmp = train_test_split(perturbs,
                                      test_size=ratios[1] + ratios[2],
                                      random_state=random_state)
        test, dev = train_test_split(tmp,
                                     test_size=ratios[2] /
                                     (ratios[1] + ratios[2]),
                                     random_state=random_state)
        return {"control": control, "train": train, "dev": dev, "test": test}
