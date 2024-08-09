# -*- coding: utf-8 -*-
# Author: TDC Team
# License: MIT
from packaging import version
import pkg_resources
"""This file contains all metadata of datasets in TDC.

Attributes:
    adme_dataset_names (list): all adme dataset names
    admet_benchmark (dict): a dictionary with key the TDC task and value a list of dataset names
    admet_metrics (dict): a dictionary with key the dataset name and value the recommended metric
    admet_splits (dict): a dictionary with key the dataset name and value the recommended split
    antibodyaff_dataset_names (list): all antibody_aff dataset names
    benchmark2id (dict): benchmark names to dataverse download ID
    benchmark2type (dict): benchmark names to file type in download format
    benchmark_names (dict): a dictionary mapping benchmark group name to each benchmark group dataset names
    bm_metric_names (dict): a dictionary mapping benchmark group name to each benchmark group metric names
    bm_split_names (dict): a dictionary mapping benchmark group name to each benchmark group split names
    catalyst_dataset_names (list): all catalyst dataset names
    category_names (dict): mapping from ML problem (1st tier) to all tasks
    crisproutcome_dataset_names (list): all crispr outcome dataset names
    dataset_list (list): total list of dataset names in TDC
    dataset_names (dict): mapping from task name to list of dataset names
    ddi_dataset_names (list): all ddi dataset names
    develop_dataset_names (list): all develop dataset names
    distribution_oracles (list): all distribution learning oracles, i.e. molecule evaluators
	docking_oracles (list): all docking oracles, i.e. RMSD
    docking_benchmark (dict): docking benchmark target names
    docking_target_info (dict): docking benchmark target pockets info
    download_oracle_names (list): oracle names that require downloading predictors
    drugres_dataset_names (list): all drugres dataset names
    drugsyn_benchmark (dict): drugcombo benchmark group targets
    drugsyn_dataset_names (list): all drugsyn dataset names
    drugsyn_metrics (dict): a dictionary with key the dataset name and value the recommended metric
    drugsyn_splits (dict):  a dictionary with key the dataset name and value the recommended split
    dti_dataset_names (list): all dti dataset names
    dti_dg_benchmark (dict): dti_dg benchmark group dataset names
    dti_dg_metrics (dict): a dictionary with key the dataset name and value the recommended metric
    dti_dg_splits (dict):  a dictionary with key the dataset name and value the recommended split
    epitope_dataset_names (list): all epitope dataset names
    evaluator_name (list): list of evaluator names
    forwardsyn_dataset_names (list): all reaction dataset names
    gda_dataset_names (list): all gda dataset names
    generation_datasets (list): all generation dataset names
    guacamol_oracle (list): list of oracles from guacamol
    hts_dataset_names (list): all hts dataset names
    meta_oracle_name (list): list of all meta oracle names
    molgenpaired_dataset_names (list): all molgenpaired dataset names
    mti_dataset_names (list): all mti dataset names
    name2id (dict): mapping from dataset names to dataverse id
    name2stats (dict): mapping from dataset names to statistics
    name2type (dict): mapping from dataset names to downloaded file format
    oracle2id (dict): mapping from oracle names to dataverse id
    oracle2type (dict): mapping from oracle names to downloaded file format
    receptor2id (dict): mapping from receptor id to dataverse id
    oracle_names (list): list of all oracle names
    paired_dataset_names (list): all paired dataset names
    paratope_dataset_names (list): all paratope dataset names
    peptidemhc_dataset_names (list): all peptidemhc dataset names
    ppi_dataset_names (list): all ppi dataset names
    property_names (list): a list of oracles that correspond to some molecular properties
    qm_dataset_names (list): all qm dataset names
    retrosyn_dataset_names (list): all retrosyn dataset names
    sdf_file_names (list): list of sdf file names
    single_molecule_dataset_names (list): all molgen dataset names
	multiple_molecule_dataset_names (list): all ligandmolgen dataset names
    synthetic_oracle_name (list): all oracle names for synthesis
    test_multi_pred_dataset_names (list): test multi pred task name
    test_single_pred_dataset_names (list): test single pred task name
    toxicity_dataset_names (list): all toxicity dataset names
    trivial_oracle_names (list): a list of oracle names for trivial oracles
    yield_dataset_names (list): all yield dataset names
"""
####################################
# test cases
test_single_pred_dataset_names = ["test_single_pred"]
test_multi_pred_dataset_names = ["test_multi_pred"]

# single_pred prediction

toxicity_dataset_names = [
    "tox21",
    "toxcast",
    "clintox",
    "herg_karim",
    "herg",
    "herg_central",
    "dili",
    "skin_reaction",
    "ames",
    "carcinogens_lagunin",
    "ld50_zhu",
]

adme_dataset_names = [
    "lipophilicity_astrazeneca", "solubility_aqsoldb",
    "hydrationfreeenergy_freesolv", "caco2_wang", "pampa_ncats",
    "approved_pampa_ncats", "hia_hou", "pgp_broccatelli", "bioavailability_ma",
    "vdss_lombardo", "cyp2c19_veith", "cyp2d6_veith", "cyp3a4_veith",
    "cyp1a2_veith", "cyp2c9_veith", "cyp2c9_substrate_carbonmangels",
    "cyp2d6_substrate_carbonmangels", "cyp3a4_substrate_carbonmangels",
    "bbb_martins", "b3db_classification", "b3db_regression", "ppbr_az",
    "half_life_obach", "clearance_hepatocyte_az", "clearance_microsome_az",
    "hlm", "rlm"
]

hts_dataset_names = [
    "hiv", "sarscov2_3clpro_diamond", "sarscov2_vitro_touret",
    "orexin1_receptor_butkiewicz", "m1_muscarinic_receptor_agonists_butkiewicz",
    "m1_muscarinic_receptor_antagonists_butkiewicz",
    "potassium_ion_channel_kir2.1_butkiewicz",
    "kcnq2_potassium_channel_butkiewicz",
    "cav3_t-type_calcium_channels_butkiewicz", "choline_transporter_butkiewicz",
    "serine_threonine_kinase_33_butkiewicz",
    "tyrosyl-dna_phosphodiesterase_butkiewicz"
]

qm_dataset_names = ["qm7", "qm7b", "qm8", "qm9"]

epitope_dataset_names = ["iedb_jespersen", "pdb_jespersen"]

paratope_dataset_names = ["sabdab_liberis"]

develop_dataset_names = ["tap", "sabdab_chen"]

# multi_pred prediction

dti_dataset_names = [
    "davis",
    "kiba",
    "bindingdb_kd",
    "bindingdb_ic50",
    "bindingdb_ki",
    "bindingdb_patent",
]

ppi_dataset_names = ["huri"]

peptidemhc_dataset_names = ["mhc2_iedb_jensen", "mhc1_iedb-imgt_nielsen"]

ddi_dataset_names = ["drugbank", "twosides"]

mti_dataset_names = ["mirtarbase"]

gda_dataset_names = ["disgenet"]

crisproutcome_dataset_names = ["leenay"]

drugres_dataset_names = [
    "gdsc1",
    "gdsc2",
]

drugsyn_dataset_names = ["oncopolypharmacology", "drugcomb"]

antibodyaff_dataset_names = ["protein_sabdab"]

yield_dataset_names = ["uspto_yields", "buchwald-hartwig"]

catalyst_dataset_names = ["uspto_catalyst"]

tcr_epi_dataset_names = ["weber", "panpep"]

trial_outcome_dataset_names = ['phase1', 'phase2', 'phase3']

proteinpeptide_dataset_names = ['brown_mdm2_ace2_12ca5']

cellxgene_dataset_names = [
    "scperturb_drug_AissaBenevolenskaya2021",
    "scperturb_drug_SrivatsanTrapnell2020_sciplex2",
    "scperturb_drug_SrivatsanTrapnell2020_sciplex3",
    "scperturb_drug_SrivatsanTrapnell2020_sciplex4",
    "scperturb_drug_ZhaoSims2021",
    "scperturb_gene_NormanWeissman2019",
    "scperturb_gene_ReplogleWeissman2022_rpe1",
    "scperturb_gene_ReplogleWeissman2022_k562_essential",
]

resource_dataset_names = [
    "opentargets_ra_data_splits",
    "opentargets_ibd_data_splits",
    "opentargets_ra_drug_evidence",
    "opentargets_ibd_drug_evidence",
    "opentargets_ra_data_splits_idx",
    "opentargets_ibd_data_splits_idx",
    "tchard_full",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-0",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-1",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-2",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-3",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-4",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-0",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-1",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-2",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-3",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-4",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-0",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-1",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-2",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-3",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-4",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-0",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-1",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-2",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-3",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-4",
    "tchard_pep_cdr3b_only_neg_assays_test-0",
    "tchard_pep_cdr3b_only_neg_assays_test-1",
    "tchard_pep_cdr3b_only_neg_assays_test-2",
    "tchard_pep_cdr3b_only_neg_assays_test-3",
    "tchard_pep_cdr3b_only_neg_assays_test-4",
    "tchard_pep_cdr3b_only_neg_assays_train-0",
    "tchard_pep_cdr3b_only_neg_assays_train-1",
    "tchard_pep_cdr3b_only_neg_assays_train-2",
    "tchard_pep_cdr3b_only_neg_assays_train-3",
    "tchard_pep_cdr3b_only_neg_assays_train-4",
    "tchard_pep_cdr3b_only_sampled_negs_test-0",
    "tchard_pep_cdr3b_only_sampled_negs_test-1",
    "tchard_pep_cdr3b_only_sampled_negs_test-2",
    "tchard_pep_cdr3b_only_sampled_negs_test-3",
    "tchard_pep_cdr3b_only_sampled_negs_test-4",
    "tchard_pep_cdr3b_only_sampled_negs_train-0",
    "tchard_pep_cdr3b_only_sampled_negs_train-1",
    "tchard_pep_cdr3b_only_sampled_negs_train-2",
    "tchard_pep_cdr3b_only_sampled_negs_train-3",
    "tchard_pep_cdr3b_only_sampled_negs_train-4",
]

resources = {
    "opentargets_dti": {
        "splits": [
            "opentargets_ra_data_splits",
            "opentargets_ibd_data_splits",
        ],
        "datasets": [
            "opentargets_ra_drug_evidence",
            "opentargets_ibd_drug_evidence",
        ],
        "all": [
            "opentargets_ra_data_splits",
            "opentargets_ibd_data_splits",
            "opentargets_ra_drug_evidence",
            "opentargets_ibd_drug_evidence",
        ],
    },
    "tchard": {
        "splits_raw": {
            "train": {
                "tchard_pep_cdr3b_only_neg_assays": {
                    0: "tchard_pep_cdr3b_only_neg_assays_train-0",
                    1: "tchard_pep_cdr3b_only_neg_assays_train-1",
                    2: "tchard_pep_cdr3b_only_neg_assays_train-2",
                    3: "tchard_pep_cdr3b_only_neg_assays_train-3",
                    4: "tchard_pep_cdr3b_only_neg_assays_train-4",
                },
                "tchard_pep_cdr3b_only_sampled_negs_train": {
                    0: "tchard_pep_cdr3b_only_sampled_negs_train-0",
                    1: "tchard_pep_cdr3b_only_sampled_negs_train-1",
                    2: "tchard_pep_cdr3b_only_sampled_negs_train-2",
                    3: "tchard_pep_cdr3b_only_sampled_negs_train-3",
                    4: "tchard_pep_cdr3b_only_sampled_negs_train-4",
                },
                "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train": {
                    0: "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-0",
                    1: "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-1",
                    2: "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-2",
                    3: "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-3",
                    4: "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-4",
                },
                "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train": {
                    0: "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-0",
                    1: "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-1",
                    2: "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-2",
                    3: "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-3",
                    4: "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-4",
                }
            },
            "test": {
                "tchard_pep_cdr3b_only_neg_assays": {
                    0: "tchard_pep_cdr3b_only_neg_assays_test-0",
                    1: "tchard_pep_cdr3b_only_neg_assays_test-1",
                    2: "tchard_pep_cdr3b_only_neg_assays_test-2",
                    3: "tchard_pep_cdr3b_only_neg_assays_test-3",
                    4: "tchard_pep_cdr3b_only_neg_assays_test-4",
                },
                "tchard_pep_cdr3b_only_sampled_negs_train": {
                    0: "tchard_pep_cdr3b_only_sampled_negs_test-0",
                    1: "tchard_pep_cdr3b_only_sampled_negs_test-1",
                    2: "tchard_pep_cdr3b_only_sampled_negs_test-2",
                    3: "tchard_pep_cdr3b_only_sampled_negs_test-3",
                    4: "tchard_pep_cdr3b_only_sampled_negs_test-4",
                },
                "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train": {
                    0: "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-0",
                    1: "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-1",
                    2: "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-2",
                    3: "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-3",
                    4: "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-4",
                },
                "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train": {
                    0: "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-0",
                    1: "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-1",
                    2: "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-2",
                    3: "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-3",
                    4: "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-4",
                }
            },
            "dev": {}  # no dev set on tchard
        },
        "all": ["tchard_full",],
        "config": {
            "Y": "label",
        }
    },
}

####################################
# generation

retrosyn_dataset_names = ["uspto50k", "uspto"]

forwardsyn_dataset_names = ["uspto"]

single_molecule_dataset_names = ["zinc", "moses", "chembl", "chembl_v29"]

multiple_molecule_dataset_names = ["dude", "pdbbind", "scpdb"]  #'crossdock',

paired_dataset_names = ["uspto50k", "uspto"]

####################################
# resource

compound_library_names = [
    "drugbank_drugs",
    "chembl_drugs",
    "broad_repurposing_hub",
    "antivirals",
]
biokg_library_names = ["hetionet"]

####################################
# oracles

#### evaluator for distribution learning, the input of __call__ is list of smiles
distribution_oracles = [
    "novelty",
    "diversity",
    "uniqueness",
    "validity",
    "fcd_distance",
    "kl_divergence",
]

docking_oracles = ["rmsd", "kabsch_rmsd", "smina"]

property_names = [
    "drd2",
    "qed",
    "logp",
    "sa",
    "gsk3b",
    "jnk3",
]

evaluator_name = [
    "roc-auc",
    "f1",
    "pr-auc",
    "precision",
    "recall",
    "accuracy",
    "mse",
    "rmse",
    "mae",
    "r2",
    "micro-f1",
    "macro-f1",
    "kappa",
    "avg-roc-auc",
    "rp@k",
    "pr@k",
    "pcc",
    "spearman",
    "range_logAUC",
]

evaluator_name.extend(distribution_oracles)
evaluator_name.extend(docking_oracles)

guacamol_oracle = [
    "rediscovery",
    "similarity",
    "median",
    "isomers",
    "mpo",
    "hop",
    "celecoxib_rediscovery",
    "troglitazone_rediscovery",
    "thiothixene_rediscovery",
    "aripiprazole_similarity",
    "albuterol_similarity",
    "mestranol_similarity",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "isomers_c11h24",
    "osimertinib_mpo",
    "fexofenadine_mpo",
    "ranolazine_mpo",
    "perindopril_mpo",
    "amlodipine_mpo",
    "sitagliptin_mpo",
    "zaleplon_mpo",
    "sitagliptin_mpo_prev",
    "zaleplon_mpo_prev",
    "median1",
    "median2",
    "valsartan_smarts",
    "deco_hop",
    "scaffold_hop",
]

####################################
# Benchmark Datasets

admet_benchmark = {
    "ADME": [
        "caco2_wang",
        "hia_hou",
        "pgp_broccatelli",
        "bioavailability_ma",
        "lipophilicity_astrazeneca",
        "solubility_aqsoldb",
        "bbb_martins",
        "ppbr_az",
        "vdss_lombardo",
        "cyp2d6_veith",
        "cyp3a4_veith",
        "cyp2c9_veith",
        "cyp2d6_substrate_carbonmangels",
        "cyp3a4_substrate_carbonmangels",
        "cyp2c9_substrate_carbonmangels",
        "half_life_obach",
        "clearance_microsome_az",
        "clearance_hepatocyte_az",
    ],
    "Tox": ["herg", "ames", "dili", "ld50_zhu"],
}

drugsyn_benchmark = {
    "Synergy": [
        "drugcomb_css",
        "drugcomb_hsa",
        "drugcomb_loewe",
        "drugcomb_bliss",
        "drugcomb_zip",
    ]
}

dti_dg_benchmark = {"DTI": ["bindingdb_patent"]}

docking_benchmark = {
    "Targets": [
        "1iep",
        "2rgp",
        "3eml",
        "3ny8",
        "4rlu",
        "4unn",
        "5mo4",
        "7l11",
        "3pbl",
    ]
}

docking_target_info = {
    "3pbl": {
        "center": (9, 22.5, 26),
        "size": (15, 15, 15)
    },
    "1iep": {
        "center": (15.61389189189189, 53.38013513513513, 15.454837837837842),
        "size": (15, 15, 15),
    },
    "2rgp": {
        "center": (16.292121212121213, 34.87081818181819, 92.0353030303030),
        "size": (15, 15, 15),
    },
    "3eml": {
        "center": (-9.063639999999998, -7.1446, 55.86259999999999),
        "size": (15, 15, 15),
    },
    "3ny8": {
        "center": (2.2488, 4.68495, 51.39820000000001),
        "size": (15, 15, 15)
    },
    "4rlu": {
        "center": (-0.7359999999999999, 22.75547368421052, -31.2368947368421),
        "size": (15, 15, 15),
    },
    "4unn": {
        "center": (5.684346153846153, 18.191769230769232, -7.37157692307692),
        "size": (15, 15, 15),
    },
    "5mo4": {
        "center": (-44.901709677419355, 20.490354838709674, 8.483354838709678),
        "size": (15, 15, 15),
    },
    "7l11": {
        "center": (-21.814812500000006, -4.216062499999999, -27.983781250000),
        "size": (15, 15, 15),
    },
}

####################################

#### Benchmark Metrics
admet_metrics = {
    "caco2_wang": "mae",
    "hia_hou": "roc-auc",
    "pgp_broccatelli": "roc-auc",
    "bioavailability_ma": "roc-auc",
    "lipophilicity_astrazeneca": "mae",
    "solubility_aqsoldb": "mae",
    "bbb_martins": "roc-auc",
    "ppbr_az": "mae",
    "vdss_lombardo": "spearman",
    "cyp2c9_veith": "pr-auc",
    "cyp2d6_veith": "pr-auc",
    "cyp3a4_veith": "pr-auc",
    "cyp2c9_substrate_carbonmangels": "pr-auc",
    "cyp3a4_substrate_carbonmangels": "roc-auc",
    "cyp2d6_substrate_carbonmangels": "pr-auc",
    "half_life_obach": "spearman",
    "clearance_hepatocyte_az": "spearman",
    "clearance_microsome_az": "spearman",
    "ld50_zhu": "mae",
    "herg": "roc-auc",
    "ames": "roc-auc",
    "dili": "roc-auc",
}

drugsyn_metrics = {
    "drugcomb_css": "mae",
    "drugcomb_hsa": "mae",
    "drugcomb_loewe": "mae",
    "drugcomb_bliss": "mae",
    "drugcomb_zip": "mae",
    "drugcomb_css_brain": "mae",
    "drugcomb_css_ovary": "mae",
    "drugcomb_css_lung": "mae",
    "drugcomb_css_skin": "mae",
    "drugcomb_css_hematopoietic_lymphoid": "mae",
    "drugcomb_css_breast": "mae",
    "drugcomb_css_prostate": "mae",
    "drugcomb_css_kidney": "mae",
    "drugcomb_css_colon": "mae",
}

dti_dg_metrics = {"bindingdb_patent": "pcc"}

#### Benchmark Splits
admet_splits = {
    "caco2_wang": "scaffold",
    "hia_hou": "scaffold",
    "pgp_broccatelli": "scaffold",
    "bioavailability_ma": "scaffold",
    "lipophilicity_astrazeneca": "scaffold",
    "solubility_aqsoldb": "scaffold",
    "bbb_martins": "scaffold",
    "ppbr_az": "scaffold",
    "vdss_lombardo": "scaffold",
    "cyp2c9_veith": "scaffold",
    "cyp2d6_veith": "scaffold",
    "cyp3a4_veith": "scaffold",
    "cyp2c9_substrate_carbonmangels": "scaffold",
    "cyp3a4_substrate_carbonmangels": "scaffold",
    "cyp2d6_substrate_carbonmangels": "scaffold",
    "half_life_obach": "scaffold",
    "clearance_hepatocyte_az": "scaffold",
    "clearance_microsome_az": "scaffold",
    "ld50_zhu": "scaffold",
    "herg": "scaffold",
    "ames": "scaffold",
    "dili": "scaffold",
}

drugsyn_splits = {
    "drugcomb_css": "combination",
    "drugcomb_hsa": "combination",
    "drugcomb_loewe": "combination",
    "drugcomb_bliss": "combination",
    "drugcomb_zip": "combination",
}

dti_dg_splits = {"bindingdb_patent": "group"}

####################################

# evaluator for single molecule, the input of __call__ is a single smiles OR list of smiles
download_oracle_names = [
    "drd2", "gsk3b", "jnk3", "fpscores", "cyp3a4_veith", "smina"
]
# download_oracle_names = ['drd2', 'gsk3b', 'jnk3', 'fpscores', 'cyp3a4_veith']
download_oracle_names = ["drd2", "gsk3b", "jnk3", "fpscores", "cyp3a4_veith"
                        ] + [
                            "drd2_current",
                            "gsk3b_current",
                            "jnk3_current",
                        ]

trivial_oracle_names = ["qed", "logp", "sa"] + guacamol_oracle
synthetic_oracle_name = ["askcos", "ibm_rxn"]
download_receptor_oracle_name = [
    "1iep_docking",
    "2rgp_docking",
    "3eml_docking",
    "3ny8_docking",
    "4rlu_docking",
    "4unn_docking",
    "5mo4_docking",
    "7l11_docking",
    "drd3_docking",
    "3pbl_docking",
    "1iep_docking_normalize",
    "2rgp_docking_normalize",
    "3eml_docking_normalize",
    "3ny8_docking_normalize",
    "4rlu_docking_normalize",
    "4unn_docking_normalize",
    "5mo4_docking_normalize",
    "7l11_docking_normalize",
    "drd3_docking_normalize",
    "3pbl_docking_normalize",
    "1iep_docking_vina",
    "2rgp_docking_vina",
    "3eml_docking_vina",
    "3ny8_docking_vina",
    "4rlu_docking_vina",
    "4unn_docking_vina",
    "5mo4_docking_vina",
    "7l11_docking_vina",
    "drd3_docking_vina",
    "3pbl_docking_vina",
]

meta_oracle_name = [
    "isomer_meta",
    "rediscovery_meta",
    "similarity_meta",
    "median_meta",
    "docking_score",
    "molecule_one_synthesis",
    "pyscreener",
]

oracle_names = (download_oracle_names + trivial_oracle_names +
                distribution_oracles + synthetic_oracle_name +
                meta_oracle_name + docking_oracles +
                download_receptor_oracle_name)

molgenpaired_dataset_names = ["qed", "drd2", "logp"]

generation_datasets = (retrosyn_dataset_names + forwardsyn_dataset_names +
                       molgenpaired_dataset_names +
                       multiple_molecule_dataset_names)
# generation
####################################

category_names = {
    "single_pred": [
        "Tox",
        "ADME",
        "HTS",
        "Epitope",
        "Develop",
        "QM",
        "Paratope",
        "Yields",
        "CRISPROutcome",
    ],
    "multi_pred": [
        "DTI", "PPI", "DDI", "PeptideMHC", "DrugRes", "AntibodyAff", "DrugSyn",
        "MTI", "GDA", "Catalyst", "TCR_Epitope_Binding", "TrialOutcome",
        "CellXGene"
    ],
    "generation": ["RetroSyn", "Reaction", "MolGen"],
}


def get_task2category():
    task2category = {}
    for i, j in category_names.items():
        for x in j:
            task2category[x] = i
    return task2category


dataset_names = {
    "Tox": toxicity_dataset_names,
    "ADME": adme_dataset_names,
    "HTS": hts_dataset_names,
    "DTI": dti_dataset_names,
    "PPI": ppi_dataset_names,
    "DDI": ddi_dataset_names,
    "RetroSyn": retrosyn_dataset_names,
    "Reaction": forwardsyn_dataset_names,
    "MolGen": single_molecule_dataset_names,
    "sbdd": multiple_molecule_dataset_names,
    "PeptideMHC": peptidemhc_dataset_names,
    "Epitope": epitope_dataset_names,
    "Develop": develop_dataset_names,
    "DrugRes": drugres_dataset_names,
    "QM": qm_dataset_names,
    "AntibodyAff": antibodyaff_dataset_names,
    "DrugSyn": drugsyn_dataset_names,
    "MTI": mti_dataset_names,
    "GDA": gda_dataset_names,
    "Paratope": paratope_dataset_names,
    "Yields": yield_dataset_names,
    "Catalyst": catalyst_dataset_names,
    "CRISPROutcome": crisproutcome_dataset_names,
    "test_single_pred": test_single_pred_dataset_names,
    "test_multi_pred": test_multi_pred_dataset_names,
    "TCREpitopeBinding": tcr_epi_dataset_names,
    "TrialOutcome": trial_outcome_dataset_names,
    "ProteinPeptide": proteinpeptide_dataset_names,
    "CellXGene": cellxgene_dataset_names,
    "Resource": resource_dataset_names,
}

benchmark_names = {
    "admet_group": admet_benchmark,
    "drugcombo_group": drugsyn_benchmark,
    "docking_group": docking_benchmark,
    "dti_dg_group": dti_dg_benchmark,
}

bm_metric_names = {
    "admet_group": admet_metrics,
    "drugcombo_group": drugsyn_metrics,
    "dti_dg_group": dti_dg_metrics,
}

bm_split_names = {
    "admet_group": admet_splits,
    "drugcombo_group": drugsyn_splits,
    "dti_dg_group": dti_dg_splits,
}

dataset_list = []
for i in dataset_names.keys():
    dataset_list = dataset_list + [i.lower() for i in dataset_names[i]]

name2type = {
    "toxcast": "tab",
    "tox21": "tab",
    "clintox": "tab",
    "lipophilicity_astrazeneca": "tab",
    "solubility_aqsoldb": "tab",
    "hydrationfreeenergy_freesolv": "tab",
    "caco2_wang": "tab",
    "pampa_ncats": "tab",
    "approved_pampa_ncats": "tab",
    "hia_hou": "tab",
    "pgp_broccatelli": "tab",
    "f20_edrug3d": "tab",
    "f30_edrug3d": "tab",
    "bioavailability_ma": "tab",
    "vd_edrug3d": "tab",
    "cyp2c19_veith": "tab",
    "cyp2d6_veith": "tab",
    "cyp3a4_veith": "tab",
    "cyp1a2_veith": "tab",
    "cyp2c9_veith": "tab",
    "cyp2c9_substrate_carbonmangels": "tab",
    "cyp2d6_substrate_carbonmangels": "tab",
    "cyp3a4_substrate_carbonmangels": "tab",
    "carcinogens_lagunin": "tab",
    "halflife_edrug3d": "tab",
    "clearance_edrug3d": "tab",
    "bbb_adenot": "tab",
    "bbb_martins": "tab",
    "b3db_classification": "tab",
    "b3db_regression": "tab",
    "ppbr_ma": "tab",
    "ppbr_edrug3d": "tab",
    "hiv": "tab",
    "sarscov2_3clpro_diamond": "tab",
    "sarscov2_vitro_touret": "tab",
    "orexin1_receptor_butkiewicz": "tab",
    "m1_muscarinic_receptor_agonists_butkiewicz": "tab",
    "m1_muscarinic_receptor_antagonists_butkiewicz": "tab",
    "potassium_ion_channel_kir2.1_butkiewicz": "tab",
    "kcnq2_potassium_channel_butkiewicz": "tab",
    "cav3_t-type_calcium_channels_butkiewicz": "tab",
    "choline_transporter_butkiewicz": "tab",
    "serine_threonine_kinase_33_butkiewicz": "tab",
    "tyrosyl-dna_phosphodiesterase_butkiewicz": "tab",
    "davis": "tab",
    "kiba": "tab",
    "bindingdb_kd": "tab",
    "bindingdb_ic50": "csv",
    "bindingdb_ki": "csv",
    "bindingdb_patent": "csv",
    "huri": "tab",
    "drugbank": "tab",
    "twosides": "csv",
    "mhc1_iedb-imgt_nielsen": "tab",
    "mhc2_iedb_jensen": "tab",
    "uspto": "csv",
    "uspto50k": "tab",
    "zinc": "tab",
    "moses": "tab",
    "chembl": "tab",
    "chembl_v29": "csv",
    "qed": "tab",
    "drd2": "tab",
    "logp": "tab",
    "drugcomb": "pkl",
    "gdsc1": "pkl",
    "gdsc2": "pkl",
    "iedb_jespersen": "pkl",
    "pdb_jespersen": "pkl",
    "qm7": "pkl",
    "qm7b": "pkl",
    "qm8": "pkl",
    "qm9": "pkl",
    "scpdb": "zip",
    "dude": "zip",
    #  'crossdock': 'zip',
    "tap": "tab",
    "sabdab_chen": "tab",
    "protein_sabdab": "csv",
    "oncopolypharmacology": "pkl",
    "mirtarbase": "csv",
    "disgenet": "csv",
    "sabdab_liberis": "pkl",
    "uspto_yields": "pkl",
    "uspto_catalyst": "csv",
    "buchwald-hartwig": "pkl",
    "hetionet": "tab",
    "herg": "tab",
    "herg_central": "tab",
    "herg_karim": "tab",
    "dili": "tab",
    "ppbr_az": "tab",
    "ames": "tab",
    "skin_reaction": "tab",
    "drugbank_drugs": "csv",
    "clearance_microsome_az": "tab",
    "clearance_hepatocyte_az": "tab",
    "half_life_obach": "tab",
    "ld50_zhu": "tab",
    "vdss_lombardo": "tab",
    "leenay": "tab",
    "test_single_pred": "tab",
    "test_multi_pred": "tab",
    "gdsc_gene_symbols": "tab",
    "weber": "tab",
    "primekg": "tab",
    "primekg_drug_feature": "tab",
    "primekg_disease_feature": "tab",
    "drug_comb_meta_data": "pkl",
    "phase1": "tab",
    "phase2": "tab",
    "phase3": "tab",
    "brown_mdm2_ace2_12ca5": "xlsx",
    "scperturb_drug_AissaBenevolenskaya2021": "h5ad",
    "scperturb_drug_SrivatsanTrapnell2020_sciplex2": "h5ad",
    "scperturb_drug_SrivatsanTrapnell2020_sciplex3": "h5ad",
    "scperturb_drug_SrivatsanTrapnell2020_sciplex4": "h5ad",
    "scperturb_drug_ZhaoSims2021": "h5ad",
    "scperturb_gene_NormanWeissman2019": "h5ad",
    "scperturb_gene_ReplogleWeissman2022_rpe1": "h5ad",
    "scperturb_gene_ReplogleWeissman2022_k562_essential": "h5ad",
    "opentargets_ra_data_splits": "json",
    "opentargets_ra_data_splits_idx": "json",
    "opentargets_ibd_data_splits": "json",
    "opentargets_ibd_data_splits_idx": "json",
    "opentargets_ra_drug_evidence": "tab",
    "opentargets_ibd_drug_evidence": "tab",
    "hlm": "tab",
    "rlm": "tab",
    "tchard_full": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-0": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-1": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-2": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-3": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-4": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-0": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-1": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-2": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-3": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-4": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-0": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-1": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-2": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-3": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-4": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-0": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-1": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-2": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-3": "tab",
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-4": "tab",
    "tchard_pep_cdr3b_only_neg_assays_test-0": "tab",
    "tchard_pep_cdr3b_only_neg_assays_test-1": "tab",
    "tchard_pep_cdr3b_only_neg_assays_test-2": "tab",
    "tchard_pep_cdr3b_only_neg_assays_test-3": "tab",
    "tchard_pep_cdr3b_only_neg_assays_test-4": "tab",
    "tchard_pep_cdr3b_only_neg_assays_train-0": "tab",
    "tchard_pep_cdr3b_only_neg_assays_train-1": "tab",
    "tchard_pep_cdr3b_only_neg_assays_train-2": "tab",
    "tchard_pep_cdr3b_only_neg_assays_train-3": "tab",
    "tchard_pep_cdr3b_only_neg_assays_train-4": "tab",
    "tchard_pep_cdr3b_only_sampled_negs_test-0": "tab",
    "tchard_pep_cdr3b_only_sampled_negs_test-1": "tab",
    "tchard_pep_cdr3b_only_sampled_negs_test-2": "tab",
    "tchard_pep_cdr3b_only_sampled_negs_test-3": "tab",
    "tchard_pep_cdr3b_only_sampled_negs_test-4": "tab",
    "tchard_pep_cdr3b_only_sampled_negs_train-0": "tab",
    "tchard_pep_cdr3b_only_sampled_negs_train-1": "tab",
    "tchard_pep_cdr3b_only_sampled_negs_train-2": "tab",
    "tchard_pep_cdr3b_only_sampled_negs_train-3": "tab",
    "tchard_pep_cdr3b_only_sampled_negs_train-4": "tab",
    "cell_tissue_mg_edgelist": "txt",
    "pinnacle_global_ppi_edgelist": "txt",
    "pinnacle_protein_embed": "pth",
    "pinnacle_labels_dict": "txt",
    "panpep": "tab",
}

name2id = {
    "bbb_adenot": 4259565,
    "bbb_martins": 4259566,
    "b3db_classification": 7878566,
    "b3db_regression": 7878567,
    "bindingdb_ic50": 4291560,
    "bindingdb_kd": 4291555,
    "bindingdb_ki": 4291556,
    "bindingdb_patent": 4724851,
    "bioavailability_ma": 4259567,
    "caco2_wang": 4259569,
    "pampa_ncats": 6695858,
    "approved_pampa_ncats": 6695857,
    "clearance_edrug3d": 4259571,
    "clintox": 4259572,
    "cyp1a2_veith": 4259573,
    "cyp2c19_veith": 4259576,
    "cyp2c9_veith": 4259577,
    "cyp2d6_veith": 4259580,
    "cyp3a4_veith": 4259582,
    "cyp2c9_substrate_carbonmangels": 4259584,
    "cyp2d6_substrate_carbonmangels": 4259578,
    "cyp3a4_substrate_carbonmangels": 4259581,
    "carcinogens_lagunin": 4259570,
    "davis": 5219748,
    "drugbank": 4139573,
    "drugcomb": 4215720,
    "f20_edrug3d": 4259586,
    "f30_edrug3d": 4259589,
    "halflife_edrug3d": 4259587,
    "hia_hou": 4259591,
    "hiv": 4259593,
    "huri": 4139567,
    "hydrationfreeenergy_freesolv": 4259594,
    "kiba": 5255037,
    "lipophilicity_astrazeneca": 4259595,
    "pgp_broccatelli": 4259597,
    "ppbr_edrug3d": 4259600,
    "ppbr_ma": 4259603,
    "sarscov2_3clpro_diamond": 4259606,
    "sarscov2_vitro_touret": 4259607,
    "orexin1_receptor_butkiewicz": 6894447,
    "m1_muscarinic_receptor_agonists_butkiewicz": 6894443,
    "m1_muscarinic_receptor_antagonists_butkiewicz": 6894446,
    "potassium_ion_channel_kir2.1_butkiewicz": 6894442,
    "kcnq2_potassium_channel_butkiewicz": 6894444,
    "cav3_t-type_calcium_channels_butkiewicz": 6894445,
    "choline_transporter_butkiewicz": 6894441,
    "serine_threonine_kinase_33_butkiewicz": 6894448,
    "tyrosyl-dna_phosphodiesterase_butkiewicz": 6894440,
    "solubility_aqsoldb": 4259610,
    "tox21": 4259612,
    "toxcast": 4259613,
    "twosides": 4139574,
    "vd_edrug3d": 4259618,
    "mhc1_iedb-imgt_nielsen": 4167073,
    "mhc2_iedb_jensen": 4167074,
    "zinc": 4170963,
    "moses": 4170962,
    "chembl": 4170965,
    "chembl_v29": 5767979,
    "qed": 4170959,
    "drd2": 4170957,
    "logp": 4170961,
    "gdsc1": 4165726,
    "gdsc2": 4165727,
    "iedb_jespersen": 4165725,
    "pdb_jespersen": 4165724,
    "qm7": 6358510,
    "qm7b": 6358512,
    "qm8": 6358513,
    "qm9": 6179310,  ### 4167112, 6175612
    #  'scpdb': None,
    #  'dude': None,
    #  'crossdock': None,
    "tap": 4167113,
    "sabdab_chen": 4167164,
    "protein_sabdab": 4167357,
    "oncopolypharmacology": 4167358,
    "mirtarbase": 4167359,
    "disgenet": 4168282,
    "sabdab_liberis": 4168425,
    "uspto50k": 4171823,
    "buchwald-hartwig": 6175640,
    "uspto_yields": 4186956,
    "uspto_catalyst": 4171574,
    "uspto": 4171642,
    "hetionet": 4201734,
    "herg": 4259588,
    "herg_central": 5740618,
    "herg_karim": 6822246,
    "dili": 4259585,
    "ppbr_az": 6413140,
    "ames": 4259564,
    "skin_reaction": 4259609,
    "clearance_microsome_az": 4266186,
    "clearance_hepatocyte_az": 4266187,
    "ld50_zhu": 4267146,
    "half_life_obach": 4266799,
    "vdss_lombardo": 4267387,
    "leenay": 4279966,
    "test_single_pred": 4832455,
    "test_multi_pred": 4832456,
    "gdsc_gene_symbols": 5255026,
    "weber": 5790963,
    "primekg": 6180626,
    "primekg_drug_feature": 6180619,
    "primekg_disease_feature": 6180618,
    "drug_comb_meta_data": 7104245,
    "phase1": 7331305,
    "phase2": 7331306,
    "phase3": 7331307,
    "brown_mdm2_ace2_12ca5": 9649623,
    "scperturb_drug_AissaBenevolenskaya2021": 9845396,
    "scperturb_drug_SrivatsanTrapnell2020_sciplex2": 9845394,
    "scperturb_drug_SrivatsanTrapnell2020_sciplex3": 9845397,
    "scperturb_drug_SrivatsanTrapnell2020_sciplex4": 9845395,
    "scperturb_drug_ZhaoSims2021": 9845393,
    "scperturb_gene_NormanWeissman2019": 10133995,
    "scperturb_gene_ReplogleWeissman2022_rpe1": 10133996,
    "scperturb_gene_ReplogleWeissman2022_k562_essential": 10134031,
    "opentargets_ra_data_splits": 10141152,
    "opentargets_ibd_data_splits": 10141151,
    "opentargets_ra_data_splits_idx": 10143574,
    "opentargets_ibd_data_splits_idx": 10143573,
    "opentargets_ra_drug_evidence": 10141153,
    "opentargets_ibd_drug_evidence": 10141154,
    "hlm": 10218426,
    "rlm": 10218425,
    "tchard_full": 10228321,
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-0": 10228304,
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-1": 10228296,
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-2": 10228328,
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-3": 10228299,
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_test-4": 10228330,
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-0": 10228331,
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-1": 10228334,
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-2": 10228324,
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-3": 10228325,
    "tchard_pep_cdr3b_cdr3a_mhc_only_neg_assays_train-4": 10228327,
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-0": 10228320,
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-1": 10228295,
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-2": 10228297,
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-3": 10228294,
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_test-4": 10228309,
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-0": 10228301,
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-1": 10228310,
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-2": 10228315,
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-3": 10228311,
    "tchard_pep_cdr3b_cdr3a_mhc_only_sampled_negs_train-4": 10228335,
    "tchard_pep_cdr3b_only_neg_assays_test-0": 10228300,
    "tchard_pep_cdr3b_only_neg_assays_test-1": 10228302,
    "tchard_pep_cdr3b_only_neg_assays_test-2": 10228305,
    "tchard_pep_cdr3b_only_neg_assays_test-3": 10228298,
    "tchard_pep_cdr3b_only_neg_assays_test-4": 10228319,
    "tchard_pep_cdr3b_only_neg_assays_train-0": 10228312,
    "tchard_pep_cdr3b_only_neg_assays_train-1": 10228317,
    "tchard_pep_cdr3b_only_neg_assays_train-2": 10228333,
    "tchard_pep_cdr3b_only_neg_assays_train-3": 10228318,
    "tchard_pep_cdr3b_only_neg_assays_train-4": 10228314,
    "tchard_pep_cdr3b_only_sampled_negs_test-0": 10228329,
    "tchard_pep_cdr3b_only_sampled_negs_test-1": 10228332,
    "tchard_pep_cdr3b_only_sampled_negs_test-2": 10228303,
    "tchard_pep_cdr3b_only_sampled_negs_test-3": 10228306,
    "tchard_pep_cdr3b_only_sampled_negs_test-4": 10228308,
    "tchard_pep_cdr3b_only_sampled_negs_train-0": 10228323,
    "tchard_pep_cdr3b_only_sampled_negs_train-1": 10228313,
    "tchard_pep_cdr3b_only_sampled_negs_train-2": 10228322,
    "tchard_pep_cdr3b_only_sampled_negs_train-3": 10228316,
    "tchard_pep_cdr3b_only_sampled_negs_train-4": 10228326,
    "cell_tissue_mg_edgelist": 10407107,
    "pinnacle_global_ppi_edgelist": 10407108,
    "pinnacle_protein_embed": 10407128,
    "pinnacle_labels_dict": 10409635,
    "panpep": 10428565,
}

oracle2type = {
    "drd2": "pkl",
    "jnk3": "pkl",
    "gsk3b": "pkl",
    "fpscores": "pkl",
    "cyp3a4_veith": "pkl",
    "smina": "static",
    "drd2_current": "pkl",
    "jnk3_current": "pkl",
    "gsk3b_current": "pkl",
}

oracle2id = {
    "drd2": 4178625,
    "gsk3b": 4170295,
    "jnk3": 4170293,
    "fpscores": 4170416,
    "cyp3a4_veith": 4411249,
    "smina": 6361665,
    "cyp3a4_veith": 4411249,
    "drd2_current": 6413411,
    "jnk3_current": 6413420,
    "gsk3b_current": 6413412,
}

benchmark2type = {
    "admet_group": "zip",
    "drugcombo_group": "zip",
    "docking_group": "zip",
    "dti_dg_group": "zip",
}

benchmark2id = {
    "admet_group": 4426004,
    "drugcombo_group": 4426002,
    "docking_group": 4554082,
    "dti_dg_group": 4742443,
}

receptor2id = {
    "1iep": [5137914, 5617659],
    "2rgp": [5137916, 5617662],
    "3eml": [5137919, 5617663],
    "3ny8": [5137915, 5617665],
    "4rlu": [5137918, 5617658],
    "4unn": [5137917, 5617661],
    "5mo4": [5137920, 5617664],
    "7l11": [5137921, 5617660],
    "3pbl": [5257195, 5617666],
}  ## 'drd3': 5137901,

sdf_file_names = {"grambow": ["Product", "Reactant", "TS"]}

name2stats = {
    "caco2_wang": 906,
    "hia_hou": 578,
    "pgp_broccatelli": 1212,
    "bioavailability_ma": 640,
    "lipophilicity_astrazeneca": 4200,
    "solubility_aqsoldb": 9982,
    "bbb_martins": 1975,
    "ppbr_az": 1797,
    "vdss_lombardo": 1130,
    "cyp2c19_veith": 12092,
    "cyp2d6_veith": 13130,
    "cyp3a4_veith": 12328,
    "cyp1a2_veith": 12579,
    "cyp2c9_veith": 12092,
    "cyp2c9_substrate_carbonmangels": 666,
    "cyp2d6_substrate_carbonmangels": 664,
    "cyp3a4_substrate_carbonmangels": 667,
    "half_life_obach": 667,
    "clearance_hepatocyte_az": 1020,
    "clearance_microsome_az": 1102,
    "ld50_zhu": 7385,
    "herg": 648,
    "ames": 7255,
    "dili": 475,
    "skin_reaction": 404,
    "carcinogens_lagunin": 278,
    "tox21": 7831,
    "clintox": 1484,
    "sarscov2_vitro_touret": 1480,
    "sarscov2_3clpro_diamond": 879,
    "hiv": 41127,
    "qm7": 7165,
    "qm7b": 7211,
    "qm8": 21747,
    "qm9": 133885,
    "uspto_yields": 853638,
    "buchwald-hartwig": 55370,
    "sabdab_liberis": 1023,
    "iedb_jespersen": 3159,
    "pdb_jespersen": 447,
    "tap": 242,
    "sabdab_chen": 2409,
    "leenay": 1521,
    "bindingdb_kd": 52284,
    "bindingdb_ki": 375032,
    "bindingdb_ic50": 991486,
    "bindingdb_patent": 243344,
    "davis": 27621,
    "kiba": 118036,
    "drugbank": 191808,
    "twosides": 4649441,
    "huri": 51813,
    "disgenet": 52476,
    "gdsc1": 177310,
    "gdsc2": 92703,
    "drugcomb": 297098,
    "oncopolypharmacology": 23052,
    "mhc1_iedb-imgt_nielsen": 185985,
    "mhc2_iedb_jensen": 134281,
    "protein_sabdab": 493,
    "mirtarbase": 400082,
    "uspto_catalyst": 721799,
    "moses": 1936962,
    "zinc": 249455,
    "chembl": 1961462,
    "uspto50k": 50036,
    "uspto": 1939253,
    "phase1": 1787,
    "phase2": 6102,
    "phase3": 4576,
}

name2idlist = {
    "dude": [6429245, 6429251],
    "scpdb": [6431629, 6431631],
}
