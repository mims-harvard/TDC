# datasets for each task

# single_pred prediction
toxicity_dataset_names = ['toxcast', 'tox21', 'clintox']

adme_dataset_names = ['lipophilicity_astrazeneca',
 'solubility_aqsoldb',
 'hydrationfreeenergy_freesolv',
 'caco2_wang',
 'hia_hou',
 'pgp_broccatelli',
 'f20_edrug3d',
 'f30_edrug3d',
 'bioavailability_ma',
 'vd_edrug3d',
 'cyp2c19_veith',
 'cyp2d6_veith',
 'cyp3a4_veith',
 'cyp1a2_veith',
 'cyp2c9_veith',
 'halflife_edrug3d',
 'clearance_edrug3d',
 'bbb_adenot',
 'bbb_martins',
 'ppbr_ma',
 'ppbr_edrug3d']

hts_dataset_names = ['hiv', 
'sarscov2_3clpro_diamond', 
'sarscov2_vitro_touret']

qm_dataset_names = ['qm7', 'qm8', 'qm9']

epitope_dataset_names = ['iedb_jespersen', 'pdb_jespersen']

####################################
# multi_pred prediction
dti_dataset_names = ['davis',
 'kiba',
 'bindingdb_kd',
 'bindingdb_ic50',
 'bindingdb_ki']

ppi_dataset_names = ['huri']

peptidemhc_dataset_names = ['mhc2_netmhciipan', 'mhc1_netmhcpan']

ddi_dataset_names = ['drugbank', 'twosides']

drugres_dataset_names = ['gdsc1', 'gdsc2']

####################################
# generation
retrosyn_dataset_names = ['uspto50k']

forwardsyn_dataset_names = ['uspto50k']

single_molecule_dataset_names = ['zinc', 'moses']

paired_dataset_names = ['uspto50k']

property_names = ['drd2', 'qed', 'logp', 'sa']
guacamol_oracle = ['celecoxib_rediscovery', 'troglitazone_rediscovery', 'thiothixene_rediscovery', \
				   'aripiprazole_similarity', 'albuterol_similarity', 'mestranol_similarity', 
				   'C11H24_isomer', 'C9H10N2O2PF2Cl_isomor', \
				   'median_molecule_1', 'median_molecule_2', \
				   'Osimertinib_MPO', 'Fexofenadine_MPO', 'Ranolazine_MPO', 'Perindopril_MPO', \
				   'Amlodipine_MPO', 'Sitagliptin_MPO', 'Zaleplon_MPO', \
				   'Valsartan_SMARTS', 'deco_hop', 'scaffold_hop']

molgenpaired_dataset_names = ['qed', 'drd2', 'logp']

generation_datasets = retrosyn_dataset_names + forwardsyn_dataset_names + molgenpaired_dataset_names 
# generation
####################################

dataset_names = {"Toxicity": toxicity_dataset_names, 
				"ADME": adme_dataset_names, 
				"HTS": hts_dataset_names, 
				"DTI": dti_dataset_names, 
				"PPI": ppi_dataset_names, 
				"DDI": ddi_dataset_names,
				"RetroSyn": retrosyn_dataset_names,
				"Reaction": forwardsyn_dataset_names, 
				"PairMolGen": molgenpaired_dataset_names,
				"PeptideMHC": peptidemhc_dataset_names,
				"Epitope": epitope_dataset_names}


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
 'huri': 'tab',
 'drugbank': 'tab',
 'twosides': 'csv',
 'mhc1_netmhcpan': 'tab',
 'mhc2_netmhciipan': 'tab',
 'uspto50k': 'csv',
 'zinc': 'csv', 
 'moses': 'csv',
 'qed': 'csv', 
 'drd2': 'csv', 
 'logp': 'csv',
 'gdsc1': 'pkl',
 'gdsc2': 'pkl',
 'iedb_jespersen': 'pkl',
 'pdb_jespersen': 'pkl'
 }

name2id = {'bbb_adenot': 4139555,
 'bbb_martins': 4165564,
 'bindingdb_ic50': 4139570,
 'bindingdb_kd': 4139577,
 'bindingdb_ki': 4139569,
 'bioavailability_ma': 4139562,
 'caco2_wang': 4139582,
 'clearance_edrug3d': 4139556,
 'clintox': 4141989,
 'cyp1a2_veith': 4139566,
 'cyp2c19_veith': 4139552,
 'cyp2c9_veith': 4139575,
 'cyp2d6_veith': 4139568,
 'cyp3a4_veith': 4139576,
 'davis': 4139572,
 'drugbank': 4139573,
 'f20_edrug3d': 4139564,
 'f30_edrug3d': 4139571,
 'halflife_edrug3d': 4139559,
 'hia_hou': 4139558,
 'hiv': 4142238,
 'huri': 4139567,
 'hydrationfreeenergy_freesolv': 4139561,
 'kiba': 4139563,
 'lipophilicity_astrazeneca': 4139549,
 'pgp_broccatelli': 4139550,
 'ppbr_edrug3d': 4139560,
 'ppbr_ma': 4139554,
 'sarscov2_3clpro_diamond': 4141988,
 'sarscov2_vitro_touret': 4141986,
 'solubility_aqsoldb': 4139579,
 'tox21': 4141990,
 'toxcast': 4142185,
 'twosides': 4139574,
 'vd_edrug3d': 4139578,
 'mhc1_netmhcpan': 4150046,
 'mhc2_netmhciipan': 4150047,
 'uspto50k': 'xxxxx',
 'zinc': 'xxxxx',
 'moses': 'xxxxx',
 'qed': 'xxxx', 
 'drd2': 'xxxx', 
 'logp': 'xxxx', 
 'gdsc1': 4165726,
 'gdsc2': 4165727,
 'iedb_jespersen': 4165725, 
 'pdb_jespersen': 4165724
 }

