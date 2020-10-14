# datasets for each task

# property prediction
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
 'bbb_molnet',
 'ppbr_ma',
 'ppbr_edrug3d']

hts_dataset_names = ['hiv', 
'sarscov2_3clpro_diamond', 
'sarscov2_vitro_touret']

qm_dataset_names = ['qm7', 'qm8', 'qm9']

# interaction prediction
dti_dataset_names = ['davis',
 'kiba',
 'bindingdb_kd',
 'bindingdb_ic50',
 'bindingdb_ki']

ppi_dataset_names = ['huri']

ddi_dataset_names = ['drugbank', 'twosides']

# generation
retrosyn_dataset_names = ['uspto50k']

forwardsyn_dataset_names = ['uspto50k']

molgenpaired_dataset_names = ['qed', 'drd2', 'logp']

dataset_names = {"Toxicity": toxicity_dataset_names, 
				"ADME": adme_dataset_names, 
				"HTS": hts_dataset_names, 
				"DTI": dti_dataset_names, 
				"PPI": ppi_dataset_names, 
				"DDI": ddi_dataset_names,
				"RetroSyn": retrosyn_dataset_names,
				"ForwardSyn": forwardsyn_dataset_names, 
				"MolGenPaired": molgenpaired_dataset_names}

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
 'bbb_molnet': 'tab',
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
 'twosides': 'tab'}

name2id = {'toxcast': 0,
 'tox21': 0,
 'clintox': 0,
 'lipophilicity_astrazeneca': 0,
 'solubility_aqsoldb': 0,
 'hydrationfreeenergy_freesolv': 0,
 'caco2_wang': 0,
 'hia_hou': 0,
 'pgp_broccatelli': 0,
 'f20_edrug3d': 0,
 'f30_edrug3d': 0,
 'bioavailability_ma': 0,
 'vd_edrug3d': 0,
 'cyp2c19_veith': 0,
 'cyp2d6_veith': 0,
 'cyp3a4_veith': 0,
 'cyp1a2_veith': 0,
 'cyp2c9_veith': 0,
 'halflife_edrug3d': 0,
 'clearance_edrug3d': 0,
 'bbb_adenot': 4139555,
 'bbb_molnet': 0,
 'ppbr_ma': 0,
 'ppbr_edrug3d': 0,
 'hiv': 0,
 'sarscov2_3clpro_diamond': 0,
 'sarscov2_vitro_touret': 0,
 'davis': 0,
 'kiba': 0,
 'bindingdb_kd': 0,
 'bindingdb_ic50': 0,
 'bindingdb_ki': 0,
 'huri': 0,
 'drugbank': 0,
 'twosides': 0}