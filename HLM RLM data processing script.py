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
    names = ['Li2022_ID', 'IUPAC', 'SMILES', 'Y', 'dataset'])

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
    names = ['Liu2015_ID', 'SMILES', 'HLM_class', 'ValidationGroup'])

HLM_dataset = pd.merge(
    left = HLM_Li2022,
    right = HLM_Liu2015[['Liu2015_ID', 'SMILES']],
    how = 'left',
    on= 'SMILES')

# Take use the Liu2015 ids for the external dataset
HLM_dataset['ID'] = np.where(
    HLM_dataset['dataset'] != 'external',
    HLM_dataset['Li2022_ID'],
    HLM_dataset['Liu2015_ID'])

HLM_dataset = HLM_dataset[['ID', 'SMILES', 'Y']]
# ID: ChEMBL IDS
# Y: [0,1] 0:active / 1:inactive

# Fill in the missing IDs
#  Three IDs are missing due to the presence of the salt form
HLM_missing = HLM_dataset[pd.isna(HLM_dataset['ID'])]
print(HLM_missing)

HLM_dataset.loc[HLM_dataset['SMILES'] == 'Oc1cc(COc2cc(Cl)cc(Cl)c2)[nH]n1', 'ID'] = 'CHEMBL1939222'
HLM_dataset.loc[HLM_dataset['SMILES'] == 'CCCCCCSc1nccnc1O[C@H]2CN3CCC2C3', 'ID'] = 'CHEMBL291339'
HLM_dataset.loc[HLM_dataset['SMILES'] == 'Cn1cc(CCN)c2C(=O)C3=C(NC=CS3(=O)=O)C(=O)c12', 'ID'] = 'CHEMBL3139464'

# Sanitize molecules with molVS
standardizer = molvs.Standardizer()
fragment_remover = molvs.fragment.FragmentRemover()
HLM_dataset['X'] = [ \
    rdkit.Chem.MolToSmiles(
        fragment_remover.remove(
        standardizer.standardize(
        rdkit.Chem.MolFromSmiles(
        smiles))))
    for smiles in HLM_dataset['SMILES']]
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
    names = ['Li2022_ID', 'IUPAC', 'SMILES', 'Y', 'dataset'])

# IDs for the external dataset can be found on PubChem AID 1508591
#  Now we are going to convert CID in RLM_Pubchem to IUPAC
#  Then we can compare the IUPAC to IUPAC in RLM_Li2022 external dataset and fill CIDs
#  Since SMEILS are mutable than IUPAC, IUPAC is a better component to compare two datasets
#  Pubchem Identifier Exchange service is used for the conversion

urllib.request.urlretrieve(
    url = 'https://pubchem.ncbi.nlm.nih.gov/rest/download/.fetch/70/1043817370403718396.txt',
    filename = 'CID_IUPAC.txt')
CID_IUPAC = pd.read_csv(
    'CID_IUPAC.txt',
    sep='\t',
    names=['CID', 'IUPAC'])
CID_IUPAC.to_csv('CID_IUPAC.csv', index= False)

RLM_dataset = pd.merge(
    left = RLM_Li2022,
    right = CID_IUPAC[['CID', 'IUPAC']],
    how = 'left',
    on= 'IUPAC')

# Take use the CIDs of CID_IUPAC for the external dataset
RLM_dataset['ID'] = np.where(
    RLM_dataset['dataset'] != 'external',
    RLM_dataset['Li2022_ID'],
    RLM_dataset['CID'])

RLM_dataset = RLM_dataset[['ID', 'SMILES', 'Y']]
# ID: ChEMBL IDS / PubChem CIDs
# Y: [0,1] 0:active/ 1:inactive

# Fill in the missing IDs
RLM_missing = RLM_dataset[pd.isna(RLM_dataset['ID'])]
print(RLM_missing)

RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(cc1)C2=Nn3c(SC2)[n+](C)c4ccccc34', 'ID'] = '780954'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Nc1nc(N)c2cc(Sc3ccc4ccccc4c3)ccc2n1', 'ID'] = '3976284'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=NNC(=S)N)C1=CCC2C3CC[C@H]4CC(O)CC[C@]4(C)C3CC[C@]12C', 'ID'] = '24761217'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1cc(N)c2cc(NC(=O)Nc3ccc4nc(C)cc(N)c4c3)ccc2n1', 'ID'] = '71166'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(Cc1ccc(O)cc1)NCC(O)c2cc(O)cc(O)c2', 'ID'] = '3343'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1cc(C=C(C#N)C(=O)c2ccc(O)c(O)c2)cc(I)c1O', 'ID'] = '3725'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC1(C)C(=O)N=C2C1=C(O)C(=O)c3ccccc23', 'ID'] = '6604934'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCCC1=C(CC)C(=N)c2ccccc2N1C', 'ID'] = '659146'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc2NC(=O)C(=Cc3c[nH]cn3)c2c1', 'ID'] = '24906268'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1ncnc2[nH]cnc12', 'ID'] = '790'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'OC(=O)c1ccc(cc1)c2oc(C=C(C#N)c3nc4ccccc4[nH]3)cc2', 'ID'] = '2844169'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1ccc(cc1)C(=O)Nc2nc3nc(cc(O)n3n2)c4ccccc4', 'ID'] = '929965'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1ccc(cc1)C(=O)Nc2nc3nc(C)cc(O)n3n2', 'ID'] = '930029'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'NC(=O)C(=O)NN=Cc1oc(cc1)[N+](=O)[O-]', 'ID'] = '18637'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'NCC1CCC(CC1)C(=O)Oc2ccc(CCC(=O)O)cc2', 'ID'] = '2680'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CN(C)[C@H]1[C@@H]2[C@@H](O)[C@@H]3C(=C)c4cccc(O)c4C(=C3C(=O)[C@]2(O)C(=C(C(=O)N)C1=O)O)O', 'ID'] = '54675785'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC[C@H]1NC(=O)[C@@H](NC(=O)c2ncccc2O)C(C)OC(=O)[C@@H](NC(=O)C3CC(=O)CCN3C(=O)[C@H](Cc4ccccc4)N(C)C(=O)[C@@H]5CCCN5C1=O)c6ccccc6', 'ID'] = '15559221'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'OC(=O)c1cccc(c1)n2cccc2C=C3NC(=O)N(C3=O)c4cccc(Cl)c4', 'ID'] = '1210330'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'O=C(NN=Cc1oc(Sc2ccccn2)cc1)c3ccncc3', 'ID'] = '830819'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1onc(NS(=O)(=O)c2ccc(NC(=O)Cc3ccc(F)c(F)c3)cc2)c1', 'ID'] = '53255396'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COC12C(COC(=O)N)C3=C(N1CC4C2N4C)C(=C(C)C(=N)C3=O)O', 'ID'] = '244989'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1cccc(C(=O)Nc2ccc(cc2)S(=O)(=O)Nc3nccs3)c1O', 'ID'] = '70701436'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Nc1ccc(CNc2ccc(cc2)S(=O)(=O)Nc3nccs3)c(O)c1', 'ID'] = '70701386'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c(Cl)cc(F)cc1CNc2ccc(cc2)S(=O)(=O)Nc3nccs3', 'ID'] = '70701395'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(C)(N)CNC(=O)CC1CCC2(CC1)OOC3(O2)C4CC5CC(CC3C5)C4', 'ID'] = '10475633'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1cccc(CN(C)c2ccc(cc2)S(=O)(=O)Nc3nccs3)c1O', 'ID'] = '70701409'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'O=S(=O)(Nc1nccs1)c2ccc(NCc3cccc4CCNc34)cc2', 'ID'] = '70701365'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(C=C2C(=O)Nc3ccccc23)cc1O', 'ID'] = '4095029'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'OC(=O)c1cccc(C=C2C(=O)Nc3ccccc23)c1', 'ID'] = '1478222'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c(Cl)cc(C=C2C(=O)Nc3c(Cl)cccc23)cc1Cl', 'ID'] = '72710613'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c(Cl)cc(C=C2C(=O)Nc3cc(Cl)ccc23)cc1Cl', 'ID'] = '72710618'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c(Cl)cc(C=C2C(=O)Nc3ccc(Cl)cc23)cc1Cl', 'ID'] = '24906292'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c(Cl)cc(C=C2C(=O)Nc3cccc(Cl)c23)cc1Cl', 'ID'] = '72710611'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c(Cl)cccc1CNc2ccc(cc2)S(=O)(=O)Nc3ccc(cc3)N4CCNCC4', 'ID'] = '70701408'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1cc(Br)ccc1CNc2ccc(cc2)S(=O)(=O)Nc3ccc(cc3)N4CCNCC4', 'ID'] = '70701402'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1cc(Cl)ccc1CNc2ccc(cc2)S(=O)(=O)Nc3ccc(cc3)N4CCNCC4', 'ID'] = '70701431'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'O=C(Nc1ccc(cc1)S(=O)(=O)Nc2nccs2)c3cc[nH]n3', 'ID'] = '70789430'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'O=C(Nc1ccc(cc1)S(=O)(=O)Nc2nccs2)c3nc[nH]n3', 'ID'] = '70789466'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1cc(Br)ccc1CNc2ccc(cc2)S(=O)(=O)Nc3ccc(cc3)C4CCNCC4', 'ID'] = '70701410'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1cc(Cl)ccc1CNc2ccc(cc2)S(=O)(=O)Nc3ccc(cc3)C4CCNCC4', 'ID'] = '70701420'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(NC(=O)c2[nH]c(C)cc2C)cc1S(=O)(=O)Nc3ccc(Br)cc3', 'ID'] = '71771109'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(NC(=O)c2c(C)cccc2C)cc1S(=O)(=O)N(C)C', 'ID'] = '71771097'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1ccc(cc1)S(=O)(=O)n2cc(c3ccc(cc3)c4ccccc4)c5ccccc25', 'ID'] = '70789481'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2cccc(c2)S(=O)(=O)Nc3ccccn3)c1C', 'ID'] = '71771102'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2cccc(c2)S(=O)(=O)Nc3ccncc3)c1C', 'ID'] = '71771081'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CN(C)S(=O)(=O)c1cccc(NC(=O)c2c(C)ccnc2C)c1', 'ID'] = '71771087'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2cccc(c2)S(=O)(=O)Nc3ccc(cc3)C#N)c1C', 'ID'] = '71771093'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)Nc1ccc(NS(=O)(=O)c2cccc(NC(=O)c3[nH]c(C)c(C(=O)C)c3C)c2)cc1', 'ID'] = '71771076'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCc1ccc(NC(=O)c2[nH]c(C)c(C(=O)C)c2C)cc1S(=O)(=O)Nc3ccccc3C#N', 'ID'] = '71771067'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(NC(=O)c2[nH]c(C)c(C(=O)C)c2C)cc1S(=O)(=O)Nc3ccccc3C#N', 'ID'] = '73424672'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(NC(=O)c2[nH]c(C)c(C(=O)C)c2C)cc1S(=O)(=O)Nc3ccc(cc3)C#N', 'ID'] = '73424673'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CN(C)CCCOc1ccc(NC(=O)c2[nH]c(C)c(C(=O)C)c2C)cc1S(=O)(=O)Nc3ccc(Br)cc3', 'ID'] = '73330286'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'OC(=O)Cc1ccc2oc(nc2c1)c3ccc(NC(=O)C=Cc4ccc(Br)cc4)c(F)c3', 'ID'] = '69906816'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CS(=O)(=O)c1ccc(cc1)c2nc(C(=O)N3CCOCC3)n4ccncc24', 'ID'] = '134791272'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COC(=O)CNC(=O)c1nc(c2ccc(cc2)S(=O)(=O)C)c3cnccn13', 'ID'] = '134790244'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CS(=O)(=O)c1ccc(cc1)c2nc(C(=O)NCC#C)n3ccncc23', 'ID'] = '134787997'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CS(=O)(=O)c1ccc(cc1)c2nc(C(=O)NCC#C)n3ccccc23', 'ID'] = '134788860'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CS(=O)(=O)c1ccc(cc1)c2nc(C(=O)N3CCOCC3)n4ccccc24', 'ID'] = '134788093'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CS(=O)(=O)c1ccc(cc1)c2nc(C(=O)NCc3cccnc3)n4ccccc24', 'ID'] = '134789059'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COC(=O)CNC(=O)c1nc(c2ccc(cc2)S(=O)(=O)C)c3ccccn13', 'ID'] = '134789772'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCc1c(C)nc2c(cnn2c1O)c3cnn(c3)c4cc(CO)ccn4', 'ID'] = '91663282'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c2CCN(Cc2nc3ccnn13)C(=O)c4ccc5ncccc5c4', 'ID'] = '87052427'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1ccc(cn1)C(=O)N2CCc3c(C2)nc4cc(nn4c3O)c5ccccc5', 'ID'] = '53183866'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c2CCN(Cc2nc3cc(nn13)c4ccccc4)C(=O)CCC(=O)N5CC(=O)Nc6ccccc56', 'ID'] = '87052512'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(C)c1c(O)n2ncc(C#N)c2nc1c3ccccc3', 'ID'] = '78426698'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c(Cl)cc(C=C2C(=O)Nc3ccccc23)cc1Cl', 'ID'] = '54505794'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'OC(=O)c1ccc(C=C2C(=O)Nc3ccccc23)cc1', 'ID'] = '3274006'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc2c(CCc3c([nH]nc23)C(=O)N4CCc5c(C4)nc6cc(nn6c5O)c7ccccc7)c1', 'ID'] = '87052503'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccccc1C(=O)Nc2nc3nc(C)cc(O)n3n2', 'ID'] = '930080'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c(Br)cc(C=C2C(=O)Nc3ccccc23)cc1Br', 'ID'] = '1228885'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1cccc2C(=Cc3cc(Cl)c(O)c(Cl)c3)C(=O)Nc12', 'ID'] = '72710609'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1cc(CNCCO)cc(Cl)c1OCc2ccc(Cl)cc2Cl', 'ID'] = '2200697'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2cccc(c2)S(=O)(=O)Nc3cccc(c3)C#N)c1C', 'ID'] = '71771101'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1onc(C)c1S(=O)(=O)N2CCC3(CC2)CC(=NO3)c4cccc(c4)C(F)(F)F', 'ID'] = '134801994'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1noc(C=Cc2ccccc2F)c1S(=O)(=O)N3CCC(CC3)C(=O)Nc4ccc(cc4)C#N', 'ID'] = '72074627'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Clc1ccc(CNc2ccc(cc2)S(=O)(=O)Nc3nccs3)cc1Cl', 'ID'] = '70701442'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'C[C@]1(CO)CN(C[C@@]1(C)CO)c2cc(C(=O)Nc3ccc4CCc5c(nn(c6ccc(F)cc6)c5c4c3)C(=O)N)c(Cl)cn2', 'ID'] = '57519531'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Brc1cccc(c1)C(=O)Nc2ncc(s2)S(=O)(=O)N3CCOCC3', 'ID'] = '71660754'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c(F)cc(C=C2C(=O)Nc3ccccc23)cc1F', 'ID'] = '72710615'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCc1ccc(NC(=O)c2[nH]c(C)c(C(=O)C)c2C)cc1S(=O)(=O)Nc3ccc(Br)cc3', 'ID'] = '71771098'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(cc1)n2nc(nc2O)C3CCN(CC3)S(=O)(=O)c4c(C)onc4C', 'ID'] = '137299452'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1ccc2NC(=O)C(=Cc3cc(Cl)c(O)c(Cl)c3)c2c1', 'ID'] = '44312236'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1ccc(cc1)S(=O)(=O)n2cc(Cc3ccc(cc3)c4ccccc4)c5ccccc25', 'ID'] = '70789527'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1cc(Cl)ccc1CNc2ccc(cc2)S(=O)(=O)Nc3ccc4ccccc4c3', 'ID'] = '70701437'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1ccc(NS(=O)(=O)c2cccc(NC(=O)c3[nH]c(C)c(C(=O)C)c3C)c2)cc1', 'ID'] = '71771078'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1nc2nc(NC(=O)c3ccc(cc3)C(C)(C)C)nn2c(O)c1Cl', 'ID'] = '6485805'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'OC1(CCN(CCCC(=O)c2ccc(F)cc2)CC1)c3cccc(c3)C(F)(F)F', 'ID'] = '5567'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1ccc(C=C2C(=O)Nc3ccccc23)cc1F', 'ID'] = '72710612'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2ccc(C)c(c2)S(=O)(=O)Nc3ccc(Br)cc3)c1C', 'ID'] = '71771092'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1cc(Cl)ccc1CNc2ccc(cc2)S(=O)(=O)Nc3cnc4ccccc4c3', 'ID'] = '70701405'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1cc(Cl)ccc1CNc2ccc(cc2)S(=O)(=O)Nc3ccc(cc3)c4ccccc4', 'ID'] = '70701417'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2cccc(c2)S(=O)(=O)Nc3ccc(Br)cc3)c1C', 'ID'] = '71771099'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1cc(C=Cc2c(F)cccc2Cl)nc3ccccc13', 'ID'] = '2966132'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2cccc(c2)S(=O)(=O)Nc3cccnc3)c1C', 'ID'] = '71771073'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2cccc(c2)S(=O)(=O)Nc3ccc(Br)c(F)c3)c1C', 'ID'] = '71771090'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1ccc(C=C2C(=O)Nc3ccccc23)cc1Cl', 'ID'] = '72710610'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'O=C([C@H](Cc1ccc(OS(=O)(=O)c2cccc3cnccc23)cc1)NS(=O)(=O)c4cccc5cnccc45)N6CCN(Cc7ccccc7)CC6', 'ID'] = '73265261'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CS(=O)(=O)N1CCC(Cn2ccc3c(nc(cc23)c4cccc5[nH]ccc45)N6CCOCC6)CC1', 'ID'] = '134809342'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'C[C@H]1CCC[C@H](O)CCCC=Cc2cc(O)cc(O)c2C(=O)O1', 'ID'] = '104847'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1ccc(C=NNC(=O)c2cc3ccccc3cc2O)cc1O', 'ID'] = '708101'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(cc1)n2nc(nc2O)C3CCN(CC3)S(=O)(=O)c4cccc(c4)C(F)(F)F', 'ID'] = '137299438'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Clc1cccc(CNc2ccc(cc2)S(=O)(=O)Nc3nccs3)c1', 'ID'] = '70701427'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CN(C)S(=O)(=O)c1cccc(NC(=O)c2c(C)c(C(=O)C)c(C)n2C)c1', 'ID'] = '71771077'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(NC(=O)c2[nH]c(C)c(C(=O)C)c2C)cc1S(=O)(=O)Nc3cccc(c3)C#N', 'ID'] = '73424674'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'OC(=O)[C@@H]1O[C@@H]1C(=O)O', 'ID'] = '2734802'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCc1ccc(NC(=O)c2[nH]c(C)c(C(=O)C)c2C)cc1S(=O)(=O)Nc3ccc(cc3)C#N', 'ID'] = '73424671'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCS(=O)(=O)Nc1ccccc1C(=O)Nc2nc(cs2)c3ccccc3', 'ID'] = '70789555'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1ccc(C=C2C(=O)Nc3ccccc23)cc1', 'ID'] = '3610132'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ncccc1c2cc3c(ccn3CC4CCN(CC4)S(=O)(=O)C)c(n2)N5CCOCC5', 'ID'] = '134806720'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(Nc1ccc(cc1)S(=O)(=O)Nc2nccs2)c3ccccc3', 'ID'] = '70701403'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1cc(ccc1CNc2ccc(cc2)S(=O)(=O)Nc3nccs3)[N+](=O)[O-]', 'ID'] = '70701388'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'FC(F)(F)c1cccc(CNc2ccc(cc2)S(=O)(=O)Nc3ccccc3)c1Cl', 'ID'] = '70701375'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(cc1OC)c2csc(n2)C3=C(N)N(Cc4ccc(Cl)cc4)CC3=O', 'ID'] = '135495242'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c(Br)cc(Cl)cc1CNc2ccc(cc2)S(=O)(=O)Nc3nccs3', 'ID'] = '70701423'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1cc(C=C2C(=O)Nc3ccccc23)ccc1O', 'ID'] = '3486422'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CN(C)S(=O)(=O)N1CCN(CC1)c2ccc(OCc3nn(C)c(C)c3c4cccc5c(CCCOc6cccc7ccccc67)c(C(=O)O)n(Cc8cccnc8)c45)cc2', 'ID'] = '73265321'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1cc(C=C2C(=O)Nc3ccccc23)cc(C)c1O', 'ID'] = '69455978'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(NC(=O)c2cccnc2C)cc1S(=O)(=O)Nc3ccc(Br)cc3', 'ID'] = '73330287'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'OCC(O)CCON=C1C(=Nc2ccccc12)c3c(O)[nH]c4ccccc34', 'ID'] = '136218975'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CO[C@@H]1C=CO[C@@]2(C)Oc3c(C)c(O)c4C(=O)C(=CC(=N[C@H]5CC[C@H](O)O[C@@H]5C)c4c3C2=O)NC(=O)C(=CC(=O)[C@@H]6C[C@@H]6[C@@H](O)[C@H](C)[C@@H](O)[C@H](C)[C@H](OC(=O)C)[C@@H]1C)C', 'ID'] = '135485450'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Clc1c[nH]nc1C2NC(=NC3=C2C(=O)CCC3)Nc4oc5ccc(cc5n4)S(=O)(=O)N6CCOCC6', 'ID'] = '71660730'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Fc1c(CNc2ccc(cc2)S(=O)(=O)Nc3ccccc3)cccc1C(F)(F)F', 'ID'] = '70701394'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COCCCOc1ccnc(CS(=O)c2nc3ccccc3[nH]2)c1C', 'ID'] = '5029'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1ccc(cc1)c2nn3c(O)c4CCCCc4nc3c2C', 'ID'] = '656272'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'N=C1C=C2Nc3ccccc3OC2=CC1=O', 'ID'] = '72725'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Clc1ccc(CC(=O)Nc2ccc(cc2)S(=O)(=O)Nc3ccccn3)cc1Cl', 'ID'] = '53255389'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CN(C)S(=O)(=O)c1cccc(NC(=O)c2c(C)cccc2C)c1', 'ID'] = '71771108'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Clc1c[nH]nc1C2NC(=NC3=C2C(=O)CCC3)Nc4oc5ccc(cc5n4)S(=O)(=O)N6CCCC6', 'ID'] = '71660747'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1cc(C=C(C#N)c2nc3ccccc3[nH]2)ccc1OCc4ccc(F)cc4', 'ID'] = '2945380'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1ccccc1C=NNc2ncccc2[N+](=O)[O-]', 'ID'] = '3091226'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1cc(C)c([nH]1)C(=O)Nc2cccc(c2)S(=O)(=O)Nc3ccc(Br)cc3', 'ID'] = '71771059'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1c(Cl)cc(C=C2C(=O)Nc3ccc(cc23)C(=O)c4occc4)cc1Cl', 'ID'] = '53239978'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Clc1ccc(Sc2oc(C=C(C#N)c3nc4ccccc4[nH]3)cc2)cc1', 'ID'] = '2942142'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(NC(=O)c2ccc[nH]2)cc1S(=O)(=O)Nc3ccc(Br)cc3', 'ID'] = '71771062'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCN(CC)CCCC(C)Nc1cc(C=Cc2ccccc2Cl)nc3cc(Cl)ccc13', 'ID'] = '170327'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCn1c(C)c(C(=O)C)c(C)c1C(=O)Nc2cccc(c2)S(=O)(=O)N(C)C', 'ID'] = '71771068'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1cc(C)c([nH]1)C(=O)Nc2cccc(c2)S(=O)(=O)N3CCCC3', 'ID'] = '71771089'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(cc1)n2nc(nc2O)C3CCCN(C3)S(=O)(=O)c4cccc(c4)C(F)(F)F', 'ID'] = '137299529'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCc1ccc(NC(=O)c2[nH]c(C)c(C(=O)C)c2C)cc1S(=O)(=O)Nc3cccc(c3)C#N', 'ID'] = '71771100'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCOC(=O)c1nnc(NS(=O)(=O)c2ccc(NC(=O)Cc3ccc(Cl)c(Cl)c3)cc2)s1', 'ID'] = '53255424'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(NC(=O)c2c(C)c(C(=O)C)c(C)n2C)cc1S(=O)(=O)Nc3ccc(Br)cc3', 'ID'] = '71771091'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCN(CC)CCO[C@H]1CC[C@]2(C)[C@H]3CC[C@@]4(C)[C@@H](CCC4=O)[C@@H]3CC=C2C1', 'ID'] = '9954083'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1cc(Cl)ccc1CNc2ccc(cc2)S(=O)(=O)Nc3cccc4ccccc34', 'ID'] = '70701439'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(NC(=O)c2c(C)cccc2C)cc1S(=O)(=O)Nc3ccc(Br)cc3', 'ID'] = '71771105'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CNC1=CC=C(Br)C=C(C(=O)C=Cc2ccc(OC)cc2OC)C1=O', 'ID'] = '71966568'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COC(=C1C(C(=C(C)N=C1C)C(=O)OCCCN2CCC(CC2)(c3ccccc3)c4ccccc4)c5cccc(c5)[N+](=O)[O-])O', 'ID'] = '101628905'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'O=C1Nc2ccccc2C1=Cc3ccc[nH]3', 'ID'] = '4625'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2cccc(c2)S(=O)(=O)Nc3c(C)cccc3C)c1C', 'ID'] = '71771095'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(NC(=O)C23CC4CC(CC(C4)C2)C3)cc1S(=O)(=O)Nc5ccc(Br)cc5', 'ID'] = '71771096'
RLM_dataset.loc[RLM_dataset['SMILES'] == '[O-][N+](=O)c1ccc(C=C2C(=O)Nc3ccccc23)cc1', 'ID'] = '2803174'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCN(CC)S(=O)(=O)c1cc(NC(=O)c2[nH]c(C)c(C(=O)C)c2C)ccc1OC', 'ID'] = '71771071'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'O=C1Nc2ccccc2C1=Cc3ccnc4ccccc34', 'ID'] = '2738696'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Fc1c(Cl)cccc1CNc2ccc(cc2)S(=O)(=O)Nc3ccccc3', 'ID'] = '70701357'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2cccc(c2)S(=O)(=O)Nc3ccccc3C)c1C', 'ID'] = '71771060'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(OC)c(c1)c2[nH]c3c(cnn3c2NC4CCCCC4)C#N', 'ID'] = '16016694'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(NC(=O)c2c(C)cccc2C)cc1S(=O)(=O)N3CCCCC3', 'ID'] = '71771085'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2cccc(c2)S(=O)(=O)Nc3ccccc3)c1C', 'ID'] = '71771075'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'N=C1C2=C(CCC2)N(Cc3ccccc3)C4=C1CCC4', 'ID'] = '659366'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'OC1=C2C(Sc3ccccc3N=C2c4ccccc14)c5occc5', 'ID'] = '3129754'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1cccc(Nc2nc(NCc3occc3)c4ccccc4n2)c1', 'ID'] = '806021'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1cccc(CNc2ccc(cc2)S(=O)(=O)Nc3cc4ccccc4s3)c1O', 'ID'] = '70701359'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1onc(C)c1Cc2oc(cc2)C(=O)N3CCc4c(C3)nc5cc(nn5c4O)c6ccccc6', 'ID'] = '87052505'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2cccc(c2)S(=O)(=O)N3CC[C@@H](F)C3)c1C', 'ID'] = '71771106'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'O=C(Nc1ccc(cc1)c2ccccc2)[C@H](Cc3ccc(OS(=O)(=O)c4cccc5cnccc45)cc3)NS(=O)(=O)c6cccc7cnccc67', 'ID'] = '73265243'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1cc(C)c([nH]1)C(=O)Nc2cccc(c2)S(=O)(=O)N3CCCCC3', 'ID'] = '71771104'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CN([C@@H](Cc1ccc(OS(=O)(=O)c2cccc3cnccc23)cc1)C(=O)N4CCN(CC4)C(=O)OCc5ccccc5)S(=O)(=O)c6cccc7cnccc67', 'ID'] = '73265249'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CN1CCN(CC1)C(c2ccccc2)c3ccccc3', 'ID'] = '6726'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(C)(C)OC(=O)N[C@@H](Cc1ccc(OS(=O)(=O)c2cccc3cnccc23)cc1)C(=O)Nc4ccc(cc4)c5ccccc5', 'ID'] = '73265257'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1ccc(Nc2nc(NCCO)c3ccccc3n2)cc1C', 'ID'] = '2226126'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1cc(Cl)ccc1CNc2ccc(cc2)S(=O)(=O)Nc3cccc4cccnc34', 'ID'] = '70701392'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'O=C(Nc1ccc(cc1)S(=O)(=O)Nc2nccs2)c3ccccn3', 'ID'] = '70789501'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1cccc(CNc2ccc(cc2)S(=O)(=O)Nc3ccccc3)c1Cl', 'ID'] = '70701379'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)[nH]c(C(=O)Nc2cccc(c2)S(=O)(=O)N3CCC(F)(F)CC3)c1C', 'ID'] = '71771064'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCc1ccc(NC(=O)c2[nH]c(C)c(C(=O)C)c2C)cc1S(=O)(=O)N3CCCCCC3', 'ID'] = '71771080'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'C(Nc1nc(Nc2ccccc2)nc3ccccc13)c4occc4', 'ID'] = '947638'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CN(C1CN(C1)S(=O)(=O)c2cccc(c2)C(F)(F)F)c3ccccc3C', 'ID'] = '134791303'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CN(C)S(=O)(=O)c1ccc2Sc3ccccc3N(CCCN4CCN(C)CC4)c2c1', 'ID'] = '9429'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCCCC(C)C=C(C)C=C(C)C(=O)NC1=CC(O)(C=CC=CC=CC(=O)NC2=C(O)CCC2=O)C3OC3C1=O', 'ID'] = '4010'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCc1ccc(cc1)C2Sc3ccccc3N=C4C2=C(O)c5ccccc45', 'ID'] = '16020356'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Oc1cc(Cl)ccc1CNc2ccc(cc2)S(=O)(=O)Nc3ccccc3', 'ID'] = '70701380'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1ccc(NC(=O)c2c(C)cccc2C)cc1S(=O)(=O)N3CCCCCC3', 'ID'] = '71771094'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CN([C@@H](Cc1ccc(OS(=O)(=O)c2cccc3cnccc23)cc1)C(=O)N4CCN(CC4)C(=O)OCc5ccccc5)C(=O)OC(C)(C)C', 'ID'] = '73265250'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCN1CCC[C@H]1CNC(=O)c2c(O)c(CC)cc(Cl)c2OC', 'ID'] = '57267'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1cc(C)c([nH]1)C(=O)Nc2cccc(c2)S(=O)(=O)N3CCCCCC3', 'ID'] = '71771061'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'Cc1cccc(C)c1C(=O)Nc2cccc(c2)S(=O)(=O)N3CCCCCC3', 'ID'] = '71771074'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(C)(C)OC(=O)N[C@@H](Cc1ccc(OS(=O)(=O)c2cccc3cnccc23)cc1)C(=O)N4CCN(Cc5ccccc5)CC4', 'ID'] = '73265264'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CN1CCCC(CC2c3ccccc3Sc4ccccc24)C1', 'ID'] = '4167'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CC(=O)c1c(C)c(C(=O)Nc2cccc(c2)S(=O)(=O)N3CCCCCC3)n(C)c1C', 'ID'] = '71771103'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COC(=C1[C@H](C(=C(C)N=C1C)C(=O)O[C@@H]2CCCN(Cc3ccccc3)C2)c4cccc(c4)[N+](=O)[O-])O', 'ID'] = '101526732'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'CCc1ccccc1n2c(N)c3c(C)nnc3nc2SCC(=O)Nc4ccc(C)cc4', 'ID'] = '135437057'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COc1c(C)c2COC(=O)c2c(O)c1CC=C(C)CCC(=O)OCCN3CCOCC3', 'ID'] = '4271'
RLM_dataset.loc[RLM_dataset['SMILES'] == 'COC(=O)[C@H](Cc1ccc(OS(=O)(=O)c2cccc3cnccc23)cc1)NS(=O)(=O)c4ccc5cnccc5c4', 'ID'] = '73330458'

# Modify the 'ID' column
#  Test and train dataset are from Chembl whereas the external dataset is from PubChem
RLM_dataset['ID'] = RLM_dataset['ID'].apply(lambda x: int(x) if isinstance(x, float) else x)
RLM_dataset.loc[3108:, 'ID'] = 'CID' + RLM_dataset.loc[3108:, 'ID'].astype(str)

# Sanitize molecules with molVS
standardizer = molvs.Standardizer()
fragment_remover = molvs.fragment.FragmentRemover()
RLM_dataset['X'] = [ \
    rdkit.Chem.MolToSmiles(
        fragment_remover.remove(
        standardizer.standardize(
        rdkit.Chem.MolFromSmiles(
        smiles))))
    for smiles in RLM_dataset['SMILES']]
problems = []
for index, row in tqdm.tqdm(RLM_dataset.iterrows()):
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

RLM_dataset[['ID', 'X', 'Y']].to_csv('RLM_dataset_final.csv', index=False)
