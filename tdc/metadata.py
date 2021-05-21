####################################
# single_pred prediction

toxicity_dataset_names = ['tox21', 'clintox', 'herg', 'dili', 'skin_reaction', 'ames', 'carcinogens_lagunin', 'ld50_zhu']

adme_dataset_names = ['lipophilicity_astrazeneca',
 'solubility_aqsoldb',
 'caco2_wang',
 'hia_hou',
 'pgp_broccatelli',
 'bioavailability_ma',
 'vdss_lombardo',
 'cyp2c19_veith',
 'cyp2d6_veith',
 'cyp3a4_veith',
 'cyp1a2_veith',
 'cyp2c9_veith',
 'cyp2c9_substrate_carbonmangels', 
 'cyp2d6_substrate_carbonmangels',
 'cyp3a4_substrate_carbonmangels', 
 'bbb_martins',
 'ppbr_az',
 'half_life_obach',
 'clearance_hepatocyte_az',
 'clearance_microsome_az']

hts_dataset_names = ['hiv', 
'sarscov2_3clpro_diamond', 
'sarscov2_vitro_touret']

qm_dataset_names = ['qm7b', 'qm8', 'qm9']

epitope_dataset_names = ['iedb_jespersen', 'pdb_jespersen']

paratope_dataset_names = ['sabdab_liberis']

develop_dataset_names = ['tap', 'sabdab_chen']

####################################
# multi_pred prediction

dti_dataset_names = ['davis',
 'kiba',
 'bindingdb_kd',
 'bindingdb_ic50',
 'bindingdb_ki',
 'bindingdb_patent']

ppi_dataset_names = ['huri']

peptidemhc_dataset_names = ['mhc2_iedb_jensen', 'mhc1_iedb-imgt_nielsen']

ddi_dataset_names = ['drugbank', 'twosides']

mti_dataset_names = ['mirtarbase']

gda_dataset_names = ['disgenet']

crisproutcome_dataset_names = ['leenay']

drugres_dataset_names = ['gdsc1', 'gdsc2']

drugsyn_dataset_names = ['oncopolypharmacology', 'drugcomb']

antibodyaff_dataset_names = ['protein_sabdab']

yield_dataset_names = ['uspto_yields', 'buchwald-hartwig']

catalyst_dataset_names = ['uspto_catalyst']


####################################
# generation


retrosyn_dataset_names = ['uspto50k', 'uspto']

forwardsyn_dataset_names = ['uspto']

single_molecule_dataset_names = ['zinc', 'moses', 'chembl']

paired_dataset_names = ['uspto50k', 'uspto']


####################################
# resource

compound_library_names = ['drugbank_drugs', 'chembl_drugs', 'broad_repurposing_hub', 'antivirals']
biokg_library_names = ['hetionet']

####################################
# oracles

#### evaluator for distribution learning, the input of __call__ is list of smiles
distribution_oracles = ['novelty', 'diversity', 'uniqueness', 'validity', 'fcd_distance', 'kl_divergence']  


property_names = ['drd2', 'qed', 'logp', 'sa', 'gsk3b', 'jnk3',]

evaluator_name = ['roc-auc', 'f1', 'pr-auc', 'precision', 'recall', \
				  'accuracy', 'mse', 'rmse', 'mae', 'r2', 'micro-f1', 'macro-f1', \
				  'kappa', 'avg-roc-auc', 'rp@k', 'pr@k', 'pcc', 'spearman']

evaluator_name.extend(distribution_oracles)

guacamol_oracle = ['rediscovery', 'similarity', 'median', 'isomers', 'mpo', 'hop', \
				   'celecoxib_rediscovery', 'troglitazone_rediscovery', 'thiothixene_rediscovery', \
				   'aripiprazole_similarity', 'albuterol_similarity', 'mestranol_similarity', 
				   'isomers_c7h8n2o2', 'isomers_c9h10n2o2pf2cl', \
				   'osimertinib_mpo', 'fexofenadine_mpo', 'ranolazine_mpo', 'perindopril_mpo', \
				   'amlodipine_mpo', 'sitagliptin_mpo', 'zaleplon_mpo', \
				   'median1', 'median2', \
				   'valsartan_smarts', 'deco_hop', 'scaffold_hop']


####################################
# Benchmark Datasets

admet_benchmark = {'ADME': ['caco2_wang', 
							'hia_hou',
							'pgp_broccatelli', 
							'bioavailability_ma',
							'lipophilicity_astrazeneca',
							'solubility_aqsoldb',
							'bbb_martins',
							'ppbr_az',
							'vdss_lombardo',
							'cyp2d6_veith',
							'cyp3a4_veith',
							'cyp2c9_veith',
							'cyp2d6_substrate_carbonmangels',
							'cyp3a4_substrate_carbonmangels',
							'cyp2c9_substrate_carbonmangels',
							'half_life_obach',
							'clearance_microsome_az',
							'clearance_hepatocyte_az'],
					'Tox':['herg',
							'ames',
							'dili',
							'ld50_zhu']
					}

drugsyn_benchmark = {'Synergy': ['drugcomb_css',
                                 'drugcomb_hsa',
                                 'drugcomb_loewe',
                                 'drugcomb_bliss',
                                 'drugcomb_zip'
                                 ]}

dti_dg_benchmark = {'DTI': ['bindingdb_patent']} 

docking_benchmark = {'Targets': ['DRD3']}

docking_target_info = {'DRD3': {'center': (9, 22.5, 26), 'size': (15, 15, 15)}
						}

####################################

#### Benchmark Metrics
admet_metrics = {'caco2_wang': 'mae',
				'hia_hou': 'roc-auc',
				'pgp_broccatelli': 'roc-auc', 
				'bioavailability_ma': 'roc-auc',
				'lipophilicity_astrazeneca': 'mae',
				'solubility_aqsoldb': 'mae',
				'bbb_martins': 'roc-auc',
				'ppbr_az': 'mae',
				'vdss_lombardo': 'spearman',
				'cyp2c9_veith': 'pr-auc',
				'cyp2d6_veith': 'pr-auc',
				'cyp3a4_veith': 'pr-auc',
				'cyp2c9_substrate_carbonmangels': 'pr-auc',
				'cyp3a4_substrate_carbonmangels': 'roc-auc',
				'cyp2d6_substrate_carbonmangels': 'pr-auc',
				'half_life_obach': 'spearman',
				'clearance_hepatocyte_az': 'spearman',
				'clearance_microsome_az': 'spearman',
				'ld50_zhu': 'mae',
				'herg': 'roc-auc',
				'ames': 'roc-auc',
				'dili': 'roc-auc'
				}

drugsyn_metrics = {'drugcomb_css': 'mae',
                   'drugcomb_hsa':'mae',
                  'drugcomb_loewe':'mae',
                  'drugcomb_bliss':'mae',
                  'drugcomb_zip':'mae',
                  'drugcomb_css_brain':'mae',
                  'drugcomb_css_ovary':'mae',
                  'drugcomb_css_lung':'mae',
                  'drugcomb_css_skin':'mae',
                  'drugcomb_css_hematopoietic_lymphoid':'mae',
                  'drugcomb_css_breast':'mae',
                  'drugcomb_css_prostate':'mae',
                  'drugcomb_css_kidney':'mae',
                  'drugcomb_css_colon':'mae',
                   }

dti_dg_metrics = {'bindingdb_patent': 'pcc'}

#### Benchmark Splits
admet_splits = {'caco2_wang': 'scaffold',
				'hia_hou': 'scaffold',
				'pgp_broccatelli': 'scaffold', 
				'bioavailability_ma': 'scaffold',
				'lipophilicity_astrazeneca': 'scaffold',
				'solubility_aqsoldb': 'scaffold',
				'bbb_martins': 'scaffold',
				'ppbr_az': 'scaffold',
				'vdss_lombardo': 'scaffold',
				'cyp2c9_veith': 'scaffold',
				'cyp2d6_veith': 'scaffold',
				'cyp3a4_veith': 'scaffold',
				'cyp2c9_substrate_carbonmangels': 'scaffold',
				'cyp3a4_substrate_carbonmangels': 'scaffold',
				'cyp2d6_substrate_carbonmangels': 'scaffold',
				'half_life_obach': 'scaffold',
				'clearance_hepatocyte_az': 'scaffold',
				'clearance_microsome_az': 'scaffold',
				'ld50_zhu': 'scaffold',
				'herg': 'scaffold',
				'ames': 'scaffold',
				'dili': 'scaffold'
				}

drugsyn_splits = {'drugcomb_css': 'combination',
                    'drugcomb_hsa': 'combination',
                    'drugcomb_loewe': 'combination',
                    'drugcomb_bliss': 'combination',
                    'drugcomb_zip': 'combination'
                    }

dti_dg_splits = {'bindingdb_patent': 'time'}

####################################

# evaluator for single molecule, the input of __call__ is a single smiles OR list of smiles
download_oracle_names = ['drd2', 'gsk3b', 'jnk3', 'fpscores', 'cyp3a4_veith']
trivial_oracle_names = ['qed', 'logp', 'sa'] + guacamol_oracle
synthetic_oracle_name = ['askcos', 'ibm_rxn']

meta_oracle_name = ['isomer_meta', 'rediscovery_meta', 'similarity_meta', 'median_meta', 'docking_score', 'molecule_one_synthesis']

oracle_names = download_oracle_names + trivial_oracle_names + distribution_oracles + synthetic_oracle_name + meta_oracle_name 

molgenpaired_dataset_names = ['qed', 'drd2', 'logp']

generation_datasets = retrosyn_dataset_names + forwardsyn_dataset_names + molgenpaired_dataset_names 
# generation
####################################

category_names = {'single_pred': ["Tox",
									"ADME",
									"HTS",
									"Epitope",
									"Develop",
									"QM",
									"Paratope",
									"Yields",
									"CRISPROutcome"],
				'multi_pred': ["DTI",
								"PPI",
								"DDI",
								"PeptideMHC",
								"DrugRes",
								"AntibodyAff",
								"DrugSyn",
								"MTI",
								"GDA",
								"Catalyst"],
				'generation': ["RetroSyn",
								"Reaction",
								"MolGen"
								]
				}

def get_task2category():
	task2category = {}
	for i, j in category_names.items():
		for x in j:
			task2category[x] = i
	return task2category

dataset_names = {"Toxicity": toxicity_dataset_names, 
				"ADME": adme_dataset_names, 
				"HTS": hts_dataset_names, 
				"DTI": dti_dataset_names, 
				"PPI": ppi_dataset_names, 
				"DDI": ddi_dataset_names,
				"RetroSyn": retrosyn_dataset_names,
				"Reaction": forwardsyn_dataset_names, 
				"MolGen": single_molecule_dataset_names,
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
				"CRISPROutcome": crisproutcome_dataset_names
				}

benchmark_names = {"admet_group": admet_benchmark,
                   "drugcombo_group": drugsyn_benchmark,
                   "docking_group": docking_benchmark,
                   "dti_dg_group": dti_dg_benchmark}

bm_metric_names = {"admet_group": admet_metrics,
                  "drugcombo_group": drugsyn_metrics,
                  "dti_dg_group": dti_dg_metrics}

bm_split_names = {"admet_group": admet_splits,
                  "drugcombo_group": drugsyn_splits,
                  "dti_dg_group": dti_dg_splits}

dataset_list = []
for i in dataset_names.keys():
    dataset_list = dataset_list + [i.lower() for i in dataset_names[i]]

name2type = {'toxcast': 'tab',
 'tox21': 'tab',
 'clintox': 'tab',
 'lipophilicity_astrazeneca': 'tab',
 'solubility_aqsoldb': 'tab',
 'hydrationfreeenergy_freesolv': 'tab',
 'caco2_wang': 'tab',
 'hia_hou': 'tab',
 'pgp_broccatelli': 'tab',
 'f20_edrug3d': 'tab',
 'f30_edrug3d': 'tab',
 'bioavailability_ma': 'tab',
 'vd_edrug3d': 'tab',
 'cyp2c19_veith': 'tab',
 'cyp2d6_veith': 'tab',
 'cyp3a4_veith': 'tab',
 'cyp1a2_veith': 'tab',
 'cyp2c9_veith': 'tab',
 'cyp2c9_substrate_carbonmangels': 'tab', 
 'cyp2d6_substrate_carbonmangels': 'tab',
 'cyp3a4_substrate_carbonmangels': 'tab', 
 'carcinogens_lagunin': 'tab', 
 'halflife_edrug3d': 'tab',
 'clearance_edrug3d': 'tab',
 'bbb_adenot': 'tab',
 'bbb_martins': 'tab',
 'ppbr_ma': 'tab',
 'ppbr_edrug3d': 'tab',
 'hiv': 'tab',
 'sarscov2_3clpro_diamond': 'tab',
 'sarscov2_vitro_touret': 'tab',
 'davis': 'tab',
 'kiba': 'tab',
 'bindingdb_kd': 'tab',
 'bindingdb_ic50': 'csv',
 'bindingdb_ki': 'csv',
 'bindingdb_patent': 'csv',
 'huri': 'tab',
 'drugbank': 'tab',
 'twosides': 'csv',
 'mhc1_iedb-imgt_nielsen': 'tab',
 'mhc2_iedb_jensen': 'tab',
 'uspto': 'csv',
 'uspto50k': 'tab',
 'zinc': 'tab', 
 'moses': 'tab',
 'chembl': 'tab',
 'qed': 'tab', 
 'drd2': 'tab', 
 'logp': 'tab',
 'drugcomb':'pkl',
 'gdsc1': 'pkl',
 'gdsc2': 'pkl',
 'iedb_jespersen': 'pkl',
 'pdb_jespersen': 'pkl',
 'qm7b': 'pkl',
 'qm8': 'pkl',
 'qm9': 'pkl',
 'tap': 'tab',
 'sabdab_chen': 'tab',
 'protein_sabdab': 'csv',
 'oncopolypharmacology': 'pkl',
 'mirtarbase': 'csv',
 'disgenet': 'csv',
 'sabdab_liberis': 'pkl',
 'uspto_yields': 'pkl', 
 'uspto_catalyst': 'csv',
 'buchwald-hartwig': 'pkl',
 'hetionet':'tab', 
 'herg': 'tab',
 'dili': 'tab',
 'ppbr_az': 'tab',
 'ames': 'tab',
 'skin_reaction': 'tab',
 'drugbank_drugs': 'csv', 
 'clearance_microsome_az': 'tab',
 'clearance_hepatocyte_az': 'tab',
 'half_life_obach': 'tab',
 'ld50_zhu': 'tab',
 'vdss_lombardo': 'tab',
 'leenay':'tab'}

name2id = {'bbb_adenot': 4259565,
 'bbb_martins': 4259566,
 'bindingdb_ic50': 4291560,
 'bindingdb_kd': 4291555,
 'bindingdb_ki': 4291556,
 'bindingdb_patent': 4724851,
 'bioavailability_ma': 4259567,
 'caco2_wang': 4259569,
 'clearance_edrug3d': 4259571,
 'clintox': 4259572,
 'cyp1a2_veith': 4259573,
 'cyp2c19_veith': 4259576,
 'cyp2c9_veith': 4259577,
 'cyp2d6_veith': 4259580,
 'cyp3a4_veith': 4259582,
 'cyp2c9_substrate_carbonmangels': 4259584,
 'cyp2d6_substrate_carbonmangels': 4259578,
 'cyp3a4_substrate_carbonmangels': 4259581, 
 'carcinogens_lagunin': 4259570,
 'davis': 4291557,
 'drugbank': 4139573,
 'drugcomb': 4215720,
 'f20_edrug3d': 4259586,
 'f30_edrug3d': 4259589,
 'halflife_edrug3d': 4259587,
 'hia_hou': 4259591,
 'hiv': 4259593,
 'huri': 4139567,
 'hydrationfreeenergy_freesolv': 4259594,
 'kiba': 4291561,
 'lipophilicity_astrazeneca': 4259595,
 'pgp_broccatelli': 4259597,
 'ppbr_edrug3d': 4259600,
 'ppbr_ma': 4259603,
 'sarscov2_3clpro_diamond': 4259606,
 'sarscov2_vitro_touret': 4259607,
 'solubility_aqsoldb': 4259610,
 'tox21': 4259612,
 'toxcast': 4259613,
 'twosides': 4139574,
 'vd_edrug3d': 4259618,
 'mhc1_iedb-imgt_nielsen': 4167073,
 'mhc2_iedb_jensen': 4167074,
 'zinc': 4170963,
 'moses': 4170962,
 'chembl': 4170965,
 'qed': 4170959, 
 'drd2': 4170957, 
 'logp': 4170961, 
 'gdsc1': 4165726,
 'gdsc2': 4165727,
 'iedb_jespersen': 4165725, 
 'pdb_jespersen': 4165724,
 'qm7b': 4167096,
 'qm8': 4167110,
 'qm9': 4167112,
 'tap': 4167113,
 'sabdab_chen': 4167164,
 'protein_sabdab': 4167357,
 'oncopolypharmacology': 4167358,
 'mirtarbase': 4167359,
 'disgenet': 4168282,
 'sabdab_liberis': 4168425,
 'uspto50k': 4171823,
 'buchwald-hartwig': 4186955,
 'uspto_yields': 4186956, 
 'uspto_catalyst': 4171574,
 'uspto': 4171642, 
 'hetionet': 4201734,
 'herg': 4259588,
 'dili': 4259585,
 'ppbr_az': 4259599,
 'ames': 4259564,
 'skin_reaction': 4259609,
 'clearance_microsome_az': 4266186,
 'clearance_hepatocyte_az': 4266187,
 'ld50_zhu': 4267146,
 'half_life_obach': 4266799,
 'vdss_lombardo': 4267387,
 'leenay':4279966 }

oracle2type = {'drd2': 'pkl', 
			   'jnk3': 'pkl', 
			   'gsk3b': 'pkl',
			   'fpscores': 'pkl', 
			   'cyp3a4_veith': 'pkl', 
			   }

oracle2id = {'drd2': 4178625,
			 'gsk3b': 4170295,
			 'jnk3': 4170293,
			 'fpscores': 4170416, 
			 'cyp3a4_veith': 4411249, 
			}

benchmark2type = {'admet_group': 'zip',
                  'drugcombo_group': 'zip',
                  'docking_group': 'zip',
                  'dti_dg_group': 'zip'}

benchmark2id = {'admet_group': 4426004,
                'drugcombo_group':  4426002,
                'docking_group': 4554082,
                'dti_dg_group': 4725171}

sdf_file_names = {
	'grambow': ['Product', 'Reactant', 'TS']
}

name2stats = {
	'caco2_wang': 906,
	'hia_hou': 578,
	'pgp_broccatelli': 1212,
	'bioavailability_ma': 640,
	'lipophilicity_astrazeneca': 4200,
	'solubility_aqsoldb': 9982,
	'bbb_martins': 1975,
	'ppbr_az': 1797,
	'vdss_lombardo': 1130,
	'cyp2c19_veith': 12092,
	'cyp2d6_veith': 13130,
	'cyp3a4_veith': 12328,
	'cyp1a2_veith': 12579,
	'cyp2c9_veith': 12092,
	'cyp2c9_substrate_carbonmangels': 666,
	'cyp2d6_substrate_carbonmangels': 664,
	'cyp3a4_substrate_carbonmangels': 667,
	'half_life_obach': 667,
	'clearance_hepatocyte_az': 1020,
	'clearance_microsome_az': 1102,
	'ld50_zhu': 7385,
	'herg': 648,
	'ames': 7255,
	'dili': 475,
	'skin_reaction': 404,
	'carcinogens_lagunin':278,
	'tox21': 7831,
	'clintox': 1484,
	'sarscov2_vitro_touret': 1480,
	'sarscov2_3clpro_diamond': 879,
	'hiv': 41127,
	'qm7b': 7211,
	'qm8': 21786,
	'qm9': 133885,
	'uspto_yields': 853638,
	'buchwald-hartwig': 55370,
	'sabdab_liberis': 1023,
	'iedb_jespersen': 3159,
	'pdb_jespersen': 447,
	'tap': 242,
	'sabdab_chen': 2409,
	'leenay': 1521,
	'bindingdb_kd': 52284,
	'bindingdb_ki': 375032,
	'bindingdb_ic50': 991486,
	'bindingdb_patent': 243344,
	'davis': 27621,
	'kiba': 118036,
	'drugbank': 191808,
	'twosides': 4649441,
	'huri': 51813,
	'disgenet': 52476,
	'gdsc1': 177310,
	'gdsc2': 92703,
	'drugcomb': 297098,
	'oncopolypharmacology': 23052,
	'mhc1_iedb-imgt_nielsen': 185985,
	'mhc2_iedb_jensen': 134281,
	'protein_sabdab': 493,
	'mirtarbase': 400082,
	'uspto_catalyst': 721799,
	'moses': 1936962,
	'zinc': 249455,
	'chembl': 1961462,
	'uspto50k': 50036,
	'uspto': 1939253
}
