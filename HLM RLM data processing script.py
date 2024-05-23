import pandas as pd
import numpy as np
import urllib.request
import tqdm
import rdkit
import molvs

# HLM

# Download SI Table 2 from
#   Longqiang Li, Zhou Lu, Guixia Liu, Yun Tang, and Weihua Lim,
#   In Silico Prediction of Human and Rat Liver Microsomal Stability via Machine Learning Methods
#   Chem. Res. Toxicol. 2022, 35, 9, 1614–1624, DOI: 10.1021/acs.chemrestox.2c00207

urllib.request.urlretrieve(
    url = "https://ndownloader.figstatic.com/files/37052237",
    filename = "Li2022_SI2.xlsx")

HLM_Li2022 = pd.read_excel(
    io = "Li2022_SI2.xlsx",
    sheet_name = "HLM_all_data",
    engine = "openpyxl",
    names = ['Li2022_ID', 'IUPAC', 'Smiles', 'Y', 'dataset'])

# IDs for the external dataset can be found in the SI from
#  Ruifeng Liu, Patric Schyman, and Anders Wallqvist
#  Critically Assessing the Predictive Power of QSAR Models for Human Liver Microsomal Stability
#  J. Chem. Inf. Model. 2015, 55, 8, 1566–1575 DOI: 10.1021/acs.jcim.5b00255

urllib.request.urlretrieve(
    url = "https://ndownloader.figstatic.com/files/3772315",
    filename = "Liu2015_SI.xlsx")

HLM_Liu2015 = pd.read_excel(
    io = "Liu2015_SI.xlsx",
    engine = "openpyxl",
    names = ['Liu2015_ID', 'Smiles', 'HLM_class', 'ValidationGroup'])

HLM_dataset = pd.merge(
    left = HLM_Li2022,
    right = HLM_Liu2015[['Liu2015_ID', 'Smiles']],
    how = 'left',
    on= 'Smiles')

# Take use the Liu2015 ids for the external dataset
HLM_dataset['ID'] = np.where(
    HLM_dataset['dataset'] != 'external',
    HLM_dataset['Li2022_ID'],
    HLM_dataset['Liu2015_ID'])

HLM_dataset = HLM_dataset[['ID', 'Smiles', 'Y']]
# ID: ChEMBL IDS
# Y: [0,1] 0:active / 1:inactive

HLM_dataset.to_csv('HLM_dataset.csv', index= False)

# Fill in the missing IDs
#  Three IDs are missing due to the presence of the salt form
missing_rows = HLM_dataset[pd.isna(HLM_dataset['ID'])]
print(missing_rows)

HLM_dataset.loc[HLM_dataset['Smiles'] == 'Oc1cc(COc2cc(Cl)cc(Cl)c2)[nH]n1', 'ID'] = 'CHEMBL1939222'
HLM_dataset.loc[HLM_dataset['Smiles'] == 'CCCCCCSc1nccnc1O[C@H]2CN3CCC2C3', 'ID'] = 'CHEMBL291339'
HLM_dataset.loc[HLM_dataset['Smiles'] == 'Cn1cc(CCN)c2C(=O)C3=C(NC=CS3(=O)=O)C(=O)c12', 'ID'] = 'CHEMBL3139464'

HLM_dataset.to_csv('HLM_dataset_filled.csv', index=False)

# Sanitize molecules with molVS
standardizer = molvs.Standardizer()
fragment_remover = molvs.fragment.FragmentRemover()
HLM_dataset['X'] = [ \
    rdkit.Chem.MolToSmiles(
        fragment_remover.remove(
        standardizer.standardize(
        rdkit.Chem.MolFromSmiles(
        smiles))))
    for smiles in HLM_dataset['Smiles']]
problems = []
for index, row in tqdm.tqdm(HLM_dataset.iterrows()):
    result = molvs.validate_smiles(row['X'])
    if len(result) == 0:
        continue
    problems.append( (row['ID'], result) )

#   Most are because it includes the salt form and/or it is not neutralized
for id, alert in problems:
    print(f"ID: {id}, problem: {alert[0]}")

# There are two types of problems in this datasets
#  1. [IsotopeValidation] Molecule contains isotope 2H
#  2. [NeutralValidation] Not an overall neutral system (+1)
# These problems can be ignored for now

HLM_dataset[['ID', 'X', 'Y']].to_csv('HLM_dataset_final.csv', index=False)

##############################################
# RLM

# Suppose that you have already downloaded SI Table 2 above (Li2022)
RLM_Li2022 = pd.read_excel(
    io = "Li2022_SI2.xlsx",
    sheet_name = "RLM_all_data",
    engine = "openpyxl",
    names = ['Li2022_ID', 'IUPAC', 'Smiles', 'Y', 'dataset'])

# IDs for the external dataset can be found on PubChem AID 1508591
urllib.request.urlretrieve(
    url = "https://pubchem.ncbi.nlm.nih.gov/assay/pcget.cgi?query=download&record_type=datatable&aid=1508591&version=2.1&response_type=save",
    filename = "AID_1508591_datatable.csv")

# Suppose that you have download 
RLM_Pubchem = pd.read_csv(
    "AID_1508591_datatable.csv",
    names = ['index', 'SID', 'CID', 'Smiles', 'Activity_outcome',
              'Activity_score', 'Activity_url', 'Assaydata_comment', 
              'Phenotype', 'Half-life (minutes)', 'Analysis Comment', 'Compound QC'],
    skiprows= 3)
   
RLM_Pubchem.loc[RLM_Pubchem['Phenotype'] == 'stable', 'Phenotype'] = 0
RLM_Pubchem.loc[RLM_Pubchem['Phenotype'] == 'unstable', 'Phenotype'] = 1

# Now we are going to convert CID in RLM_Pubchem to IUPAC
#  Then we can compare the IUPAC to IUPAC in RLM_Li2022 external dataset and fill CIDs
#  Since SMEILS are mutable than IUPAC, IUPAC is easier to compare
#  Pubchem Identifier Exchange service is used for the conversion

urllib.request.urlretrieve(
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/download/.fetch/3/1453072027186875570.txt',
    filename = 'CID_IUPAC.txt')
CID_IUPAC = pd.read_table(
    'CID_IUPAC.txt',
    sep='\t',
    names=['CID', 'IUPAC']
)
CID_IUPAC.to_csv('CID_IUPAC.txt', index= False)

# Fill in the missing IDs
missing_rows = CID_IUPAC[pd.isna(CID_IUPAC['IUPAC'])]
print(missing_rows)

RLM_dataset = pd.merge(
    left = RLM_Li2022,
    right = CID_IUPAC[['CID', 'IUPAC']],
    how = 'left',
    on= 'IUPAC')

RLM_dataset = RLM_dataset.rename(columns={'CID':'ID', 'Smiles': 'X'})
RLM_dataset = RLM_dataset[['ID', 'X', 'Y']]
# ID: ChEMBL IDS
# Y: [0,1] 0:active/ 1:inactive

RLM_dataset.to_csv('RLM_dataset.csv', index= False)