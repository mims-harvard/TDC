import pandas as pd
import numpy as np
import os, sys, json
import warnings
from packaging import version
import pkg_resources

warnings.filterwarnings("ignore")

from .utils import fuzzy_search, oracle_load, receptor_load
from .metadata import (
    download_oracle_names,
    oracle_names,
    distribution_oracles,
    docking_oracles,
    download_receptor_oracle_name,
    docking_target_info,
)

SKLEARN_VERSION = version.parse(
    pkg_resources.get_distribution("scikit-learn").version)


def _normalize_docking_score(raw_score):
    return 1 / (1 + np.exp((raw_score + 7.5)))


class Oracle:
    """the oracle class to retrieve any oracle given by query name

    Args:
        name (str): the name of the oracle
        target_smiles (None, optional): target smiles for some meta-oracles
        num_max_call (None, optional): number of maximum calls for oracle, used by docking group
        **kwargs: additional parameters for some oracles
    """

    def __init__(self, name, target_smiles=None, num_max_call=None, **kwargs):
        """Summary"""
        self.target_smiles = target_smiles
        self.kwargs = kwargs
        self.normalize = lambda x: x
        name = fuzzy_search(name, oracle_names)
        if name == "drd3_docking":
            name = "3pbl_docking"
        if name == "drd3_docking_normalize":
            name = "3pbl_docking_normalize"
        if name in download_oracle_names:
            if name in ["jnk3", "gsk3b", "drd2"]:
                if SKLEARN_VERSION >= version.parse("0.24.0"):
                    name += "_current"
            ### download
            ##### e.g., jnk, gsk, drd2, ...
            self.name = oracle_load(name)
        elif name in download_receptor_oracle_name:
            ## '1iep_docking', '2rgp_docking', '7l11_docking', 'drd3_docking', '3pbl_docking',
            pdbid = name.split("_")[0]
            self.name = receptor_load(pdbid)
            self.pdbid = self.name
            self.name += "_docking"
            if "normalize" in name:
                self.name += "_normalize"
                self.normalize = _normalize_docking_score
        else:
            self.name = name
        self.evaluator_func = None
        self.assign_evaluator()
        self.num_called = 0

        if num_max_call is not None:
            self.num_max_call = num_max_call
        else:
            self.num_max_call = None

    def assign_evaluator(self):
        """assign the specific oracle function given by query oracle name"""
        self.default_property = 0.0
        if self.name == "logp":
            from .chem_utils import penalized_logp

            self.evaluator_func = penalized_logp
        elif self.name == "qed":
            from .chem_utils import qed

            self.evaluator_func = qed
        # elif self.name == 'drd2':
        elif "drd2" in self.name:
            from .chem_utils import drd2

            self.evaluator_func = drd2
        elif self.name == "cyp3a4_veith":
            from .chem_utils import cyp3a4_veith

            self.evaluator_func = cyp3a4_veith
        elif self.name == "sa":
            from .chem_utils import SA

            self.evaluator_func = SA
        # elif self.name == 'gsk3b':
        elif "gsk3b" in self.name:
            from .chem_utils import gsk3b

            oracle_object = gsk3b
            self.evaluator_func = oracle_object
        # elif self.name == 'jnk3':
        elif "jnk3" in self.name:
            from .chem_utils import jnk3

            oracle_object = jnk3()
            self.evaluator_func = oracle_object
        elif self.name == "similarity_meta":
            from .chem_utils import similarity_meta

            self.evaluator_func = similarity_meta(
                target_smiles=self.target_smiles, **self.kwargs)
        elif self.name == "rediscovery_meta":
            from .chem_utils import rediscovery_meta

            self.evaluator_func = rediscovery_meta(
                target_smiles=self.target_smiles, **self.kwargs)
        elif self.name == "isomer_meta":
            from .chem_utils import isomer_meta

            self.evaluator_func = isomer_meta(target_smiles=self.target_smiles,
                                              **self.kwargs)
        elif self.name == "median_meta":
            from .chem_utils import median_meta

            self.evaluator_func = median_meta(
                target_smiles_1=self.target_smiles[0],
                target_smiles_2=self.target_smiles[1],
                **self.kwargs)
        elif self.name == "rediscovery":
            from .chem_utils import (
                celecoxib_rediscovery,
                troglitazone_rediscovery,
                thiothixene_rediscovery,
            )

            self.evaluator_func = {
                "Celecoxib": celecoxib_rediscovery,
                "Troglitazone": troglitazone_rediscovery,
                "Thiothixene": thiothixene_rediscovery,
            }
        elif self.name == "celecoxib_rediscovery":
            from .chem_utils import celecoxib_rediscovery

            self.evaluator_func = celecoxib_rediscovery
        elif self.name == "troglitazone_rediscovery":
            from .chem_utils import troglitazone_rediscovery

            self.evaluator_func = troglitazone_rediscovery
        elif self.name == "thiothixene_rediscovery":
            from .chem_utils import thiothixene_rediscovery

            self.evaluator_func = thiothixene_rediscovery
        elif self.name == "similarity":
            from .chem_utils import (
                aripiprazole_similarity,
                albuterol_similarity,
                mestranol_similarity,
            )

            self.evaluator_func = {
                "Aripiprazole": aripiprazole_similarity,
                "Albuterol": albuterol_similarity,
                "Mestranol": mestranol_similarity,
            }
        elif self.name == "aripiprazole_similarity":
            from .chem_utils import aripiprazole_similarity

            self.evaluator_func = aripiprazole_similarity
        elif self.name == "albuterol_similarity":
            from .chem_utils import albuterol_similarity

            self.evaluator_func = albuterol_similarity
        elif self.name == "mestranol_similarity":
            from .chem_utils import mestranol_similarity

            self.evaluator_func = mestranol_similarity
        elif self.name == "median":
            from .chem_utils import median1, median2

            self.evaluator_func = {"Median 1": median1, "Median 2": median2}
        elif self.name == "median1":
            from .chem_utils import median1

            self.evaluator_func = median1
        elif self.name == "median2":
            from .chem_utils import median2

            self.evaluator_func = median2
        elif self.name == "mpo":
            from .chem_utils import (
                osimertinib_mpo,
                fexofenadine_mpo,
                ranolazine_mpo,
                perindopril_mpo,
                amlodipine_mpo,
                sitagliptin_mpo,
                zaleplon_mpo,
            )

            self.evaluator_func = {
                "Osimertinib": osimertinib_mpo,
                "Fexofenadine": fexofenadine_mpo,
                "Ranolazine": ranolazine_mpo,
                "Perindopril": perindopril_mpo,
                "Amlodipine": amlodipine_mpo,
                "Sitagliptin": sitagliptin_mpo,
                "Zaleplon": zaleplon_mpo,
            }
        elif self.name == "osimertinib_mpo":
            from .chem_utils import osimertinib_mpo

            self.evaluator_func = osimertinib_mpo
        elif self.name == "fexofenadine_mpo":
            from .chem_utils import fexofenadine_mpo

            self.evaluator_func = fexofenadine_mpo
        elif self.name == "ranolazine_mpo":
            from .chem_utils import ranolazine_mpo

            self.evaluator_func = ranolazine_mpo
        elif self.name == "perindopril_mpo":
            from .chem_utils import perindopril_mpo

            self.evaluator_func = perindopril_mpo
        elif self.name == "amlodipine_mpo":
            from .chem_utils import amlodipine_mpo

            self.evaluator_func = amlodipine_mpo
        elif self.name == "sitagliptin_mpo_prev":
            from .chem_utils import sitagliptin_mpo_prev

            self.evaluator_func = sitagliptin_mpo_prev
        elif self.name == "sitagliptin_mpo":
            from .chem_utils import sitagliptin_mpo

            self.evaluator_func = sitagliptin_mpo
        elif self.name == "zaleplon_mpo_prev":
            from .chem_utils import zaleplon_mpo_prev

            self.evaluator_func = zaleplon_mpo_prev
        elif self.name == "zaleplon_mpo":
            from .chem_utils import zaleplon_mpo

            self.evaluator_func = zaleplon_mpo
        elif self.name == "valsartan_smarts":
            from .chem_utils import valsartan_smarts

            self.evaluator_func = valsartan_smarts
        elif self.name == "hop":
            from .chem_utils import deco_hop, scaffold_hop

            self.evaluator_func = {
                "Deco Hop": deco_hop,
                "Scaffold Hop": scaffold_hop
            }
        elif self.name == "deco_hop":
            from .chem_utils import deco_hop

            self.evaluator_func = deco_hop
        elif self.name == "scaffold_hop":
            from .chem_utils import scaffold_hop

            self.evaluator_func = scaffold_hop
        elif self.name == "isomers_c7h8n2o2":
            from .chem_utils import isomers_c7h8n2o2

            self.evaluator_func = isomers_c7h8n2o2
        elif self.name == "isomers_c9h10n2o2pf2cl":
            from .chem_utils import isomers_c9h10n2o2pf2cl

            self.evaluator_func = isomers_c9h10n2o2pf2cl
        elif self.name == "isomers_c11h24":
            from .chem_utils import isomers_c11h24

            self.evaluator_func = isomers_c11h24
        elif self.name == "isomers":
            from .chem_utils import isomers_c7h8n2o2, isomers_c9h10n2o2pf2cl

            self.evaluator_func = {
                "c7h8n2o2": isomers_c7h8n2o2,
                "c9h10n2o2pf2cl": isomers_c9h10n2o2pf2cl,
            }
        elif self.name == "askcos":  #### synthetic analysis
            from .chem_utils import askcos

            self.evaluator_func = askcos
        elif self.name == "ibm_rxn":
            from .chem_utils import ibm_rxn

            self.evaluator_func = ibm_rxn
        elif self.name == "molecule_one_synthesis":
            from .chem_utils import molecule_one_retro

            self.evaluator_func = molecule_one_retro(**self.kwargs)
        elif self.name == "pyscreener":
            from .chem_utils import PyScreener_meta

            self.evaluator_func = PyScreener_meta(**self.kwargs)
        elif self.name == "docking_score":
            from .chem_utils import Vina_smiles

            self.evaluator_func = Vina_smiles(**self.kwargs)
        elif self.name == "drd3_docking_vina" or self.name == "3pbl_docking_vina":

            from .chem_utils import Vina_smiles

            pdbid = "3pbl"
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = Vina_smiles(
                receptor_pdbqt_file="./oracle/" + pdbid + ".pdbqt",
                center=center,
                box_size=boxsize,
            )

        elif (self.name == "drd3_docking" or self.name == "3pbl_docking" or
              self.name == "drd3_docking_normalize" or
              self.name == "3pbl_docking_normalize"):

            from .chem_utils import PyScreener_meta

            pdbid = "3pbl"
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = PyScreener_meta(
                receptor_pdb_file="./oracle/" + pdbid + ".pdb",
                box_center=center,
                box_size=boxsize,
            )

        elif self.name == "1iep_docking_vina":
            from .chem_utils import Vina_smiles

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = Vina_smiles(
                receptor_pdbqt_file="./oracle/" + pdbid + ".pdbqt",
                center=center,
                box_size=boxsize,
            )
        elif self.name == "1iep_docking" or self.name == "1iep_docking_normalize":
            from .chem_utils import PyScreener_meta

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = PyScreener_meta(
                receptor_pdb_file="./oracle/" + pdbid + ".pdb",
                box_center=center,
                box_size=boxsize,
            )
        elif self.name == "2rgp_docking_vina":
            from .chem_utils import Vina_smiles

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = Vina_smiles(
                receptor_pdbqt_file="./oracle/" + pdbid + ".pdbqt",
                center=center,
                box_size=boxsize,
            )
        elif self.name == "2rgp_docking" or self.name == "2rgp_docking_normalize":
            from .chem_utils import PyScreener_meta

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = PyScreener_meta(
                receptor_pdb_file="./oracle/" + pdbid + ".pdb",
                box_center=center,
                box_size=boxsize,
            )
        elif self.name == "3eml_docking_vina":
            from .chem_utils import Vina_smiles

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = Vina_smiles(
                receptor_pdbqt_file="./oracle/" + pdbid + ".pdbqt",
                center=center,
                box_size=boxsize,
            )
        elif self.name == "3eml_docking" or self.name == "3eml_docking_normalize":
            from .chem_utils import PyScreener_meta

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = PyScreener_meta(
                receptor_pdb_file="./oracle/" + pdbid + ".pdb",
                box_center=center,
                box_size=boxsize,
            )
        elif self.name == "3ny8_docking_vina":
            from .chem_utils import Vina_smiles

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = Vina_smiles(
                receptor_pdbqt_file="./oracle/" + pdbid + ".pdbqt",
                center=center,
                box_size=boxsize,
            )
        elif self.name == "3ny8_docking" or self.name == "3ny8_docking_normalize":
            from .chem_utils import PyScreener_meta

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = PyScreener_meta(
                receptor_pdb_file="./oracle/" + pdbid + ".pdb",
                box_center=center,
                box_size=boxsize,
            )
        elif self.name == "4rlu_docking_vina":
            from .chem_utils import Vina_smiles

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = Vina_smiles(
                receptor_pdbqt_file="./oracle/" + pdbid + ".pdbqt",
                center=center,
                box_size=boxsize,
            )
        elif self.name == "4rlu_docking" or self.name == "4rlu_docking_normalize":
            from .chem_utils import PyScreener_meta

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = PyScreener_meta(
                receptor_pdb_file="./oracle/" + pdbid + ".pdb",
                box_center=center,
                box_size=boxsize,
            )
        elif self.name == "4unn_docking_vina":
            from .chem_utils import Vina_smiles

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = Vina_smiles(
                receptor_pdbqt_file="./oracle/" + pdbid + ".pdbqt",
                center=center,
                box_size=boxsize,
            )
        elif self.name == "4unn_docking" or self.name == "4unn_docking_normalize":
            from .chem_utils import PyScreener_meta

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = PyScreener_meta(
                receptor_pdb_file="./oracle/" + pdbid + ".pdb",
                box_center=center,
                box_size=boxsize,
            )
        elif self.name == "5mo4_docking_vina":
            from .chem_utils import Vina_smiles

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = Vina_smiles(
                receptor_pdbqt_file="./oracle/" + pdbid + ".pdbqt",
                center=center,
                box_size=boxsize,
            )
        elif self.name == "5mo4_docking" or self.name == "5mo4_docking_normalize":
            from .chem_utils import PyScreener_meta

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = PyScreener_meta(
                receptor_pdb_file="./oracle/" + pdbid + ".pdb",
                box_center=center,
                box_size=boxsize,
            )
        elif self.name == "7l11_docking_vina":
            from .chem_utils import Vina_smiles

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = Vina_smiles(
                receptor_pdbqt_file="./oracle/" + pdbid + ".pdbqt",
                center=center,
                box_size=boxsize,
            )
        elif self.name == "7l11_docking" or self.name == "7l11_docking_normalize":
            from .chem_utils import PyScreener_meta

            pdbid = self.name.split("_")[0]
            center = docking_target_info[pdbid]["center"]
            boxsize = docking_target_info[pdbid]["size"]
            self.evaluator_func = PyScreener_meta(
                receptor_pdb_file="./oracle/" + pdbid + ".pdb",
                box_center=center,
                box_size=boxsize,
            )
        # elif self.name == '3pbl_docking':
        # 	from .chem_utils import Vina_smiles
        # 	pdbid = self.name.split('_')[0]
        # 	center = docking_target_info[pdbid]['center']
        # 	boxsize = docking_target_info[pdbid]['size']
        # 	self.evaluator_func = Vina_smiles(receptor_pdbqt_file='./oracle/'+pdbid+'.pdbqt',
        # 									  center = center,
        # 									  box_size = boxsize)
        elif self.name == "uniqueness":
            from .chem_utils import uniqueness

            self.evaluator_func = uniqueness
        elif self.name == "validity":
            from .chem_utils import validity

            self.evaluator_func = validity
        elif self.name == "diversity":
            from .chem_utils import diversity

            self.evaluator_func = diversity
        elif self.name == "novelty":
            from .chem_utils import novelty

            self.evaluator_func = novelty
        elif self.name == "fcd_distance":
            from .chem_utils import fcd_distance

            self.evaluator_func = fcd_distance
        elif self.name == "kl_divergence":
            from .chem_utils import kl_divergence

            self.evaluator_func = kl_divergence

        elif self.name == "smina":
            from .chem_utils import smina

            self.evaluator_func = smina

        else:
            return

    def __call__(self, *args, **kwargs):
        """call the oracle function on SMILES to genenerate scores

        Args:
            *args: a list of SMILES/a string of SMILES
            **kwargs: additional parameters for some oracles

        Returns:
            float/list: the oracle score(s) for a single/list of SMILES

        Raises:
            ValueError: reached number of maximum calls if set and has queries the oracle more than the internal call counters
        """
        if self.name in distribution_oracles:
            ## 'novelty', 'diversity', 'uniqueness', 'validity', 'fcd_distance', 'kl_divergence'
            return self.evaluator_func(*args, **kwargs)

        if self.name in docking_oracles:
            return self.evaluator_func(*args, **kwargs)

        from rdkit import Chem

        smiles_lst = args[0]
        if self.name == "molecule_one_synthesis":
            return self.evaluator_func(*args, **kwargs)

        if type(smiles_lst) == list:
            nonvalid_smiles_idx_lst, valid_smiles_lst, valid_smiles_idx_lst = [], [], []
            NN = len(smiles_lst)
            for idx, smiles in enumerate(smiles_lst):
                if Chem.MolFromSmiles(smiles) == None:
                    nonvalid_smiles_idx_lst.append(idx)
                else:
                    valid_smiles_idx_lst.append(idx)
                    valid_smiles_lst.append(smiles)
            smiles_lst = valid_smiles_lst

            self.num_called += len(smiles_lst)
            if self.num_max_call is not None:
                if self.num_max_call < self.num_called:
                    self.num_called -= len(smiles_lst)
                    raise ValueError(
                        "The maximum number of evaluator call is reached! The maximum is: "
                        + str(self.num_max_call) +
                        ". The current requested call (plus accumulated calls) is: "
                        + str(self.num_called + len(smiles_lst)))

            #### evaluator for single molecule,
            #### the input of __call__ is a single smiles OR list of smiles
            if isinstance(self.evaluator_func, dict):
                all_ = {}
                for i, fct in self.evaluator_func.items():
                    results_lst = []
                    for smiles in smiles_lst:
                        results_lst.append(fct(smiles, *(args[1:]), **kwargs))
                    all_[i] = results_lst
                return all_
            else:
                results_lst = []

                if not self.name == "docking_score":
                    for smiles in smiles_lst:
                        results_lst.append(
                            self.normalize(
                                self.evaluator_func(smiles, *(args[1:]),
                                                    **kwargs)))
                else:
                    results_lst = []
                    for smiles in smiles_lst:
                        try:
                            results = self.evaluator_func([smiles], *(args[1:]),
                                                          **kwargs)
                            results = results[0]
                        except:
                            results = self.default_property
                        results_lst.append(results)
                    # results_lst = self.evaluator_func(smiles_lst, *(args[1:]), **kwargs)
                    results_lst = [self.normalize(i) for i in results_lst]
                all_results_lst = [self.default_property for i in range(NN)]
                for idx, result in zip(valid_smiles_idx_lst, results_lst):
                    all_results_lst[idx] = result
                return all_results_lst

        else:  ### a string of SMILES
            if Chem.MolFromSmiles(smiles_lst) == None:
                return self.default_property

            self.num_called += 1
            if self.num_max_call is not None:
                if self.num_max_call < self.num_called:
                    self.num_called -= 1
                    raise ValueError(
                        "The maximum number of evaluator call is reached! The maximum is: "
                        + str(self.num_max_call) +
                        ". The current requested call (plus accumulated calls) is: "
                        + str(self.num_called + 1))

            ## a single smiles
            if type(self.evaluator_func) == dict:
                all_ = {}
                for i, fct in self.evaluator_func.items():
                    all_[i] = fct(*args, **kwargs)
                return all_
            else:
                try:
                    score = self.evaluator_func(*args, **kwargs)
                except:
                    score = self.default_property
                return self.normalize(score)
                # return self.normalize(self.evaluator_func(*args, **kwargs))
