<p align="center"><img src="fig/logo.png" alt="logo" width="600px" /></p>

This repository hosts Therapeutics Data Commons (TDC), an open, user-friendly and extensive dataset hub for medicinal machine learning tasks. So far, it includes more than 50+ datasets for 20+ tasks (ranging from target identification, virtual screening, QSAR to manufacturing, safety survellience and etc) in most of the therapeutics development stages. 

[Project Website](tdc.hms.harvard.edu)

## Features

- From Bench to Bedside: covers 50+ datasets for 20+ tasks in most of the therapeutics development stages.
- Ready-to-use: the dataset is processed into machine learning ready format. The output can directly feed into prediction library such as scikit-learn and DeepPurpose. 
- User-friendly: 3 lines of codes to access the dataset and hassle-free installations.
- Helper functions: TDC supports various useful functions such as molecule generation oracles, conversion to DGL/PyG graph for interaction data, cold/scaffold split, label distribution visualization, and so much more! 
- Benchmark: provides a benchmark mode for fair comparison. A leaderboard will be released!
- Community-driven effort: the programming framework is designed to easily add more datasets and tasks.

## Example
```python
from tdc.single_pred import ADME
data = ADME(name = 'HIA_Hou')
# scaffold split using benchmark seed
split = data.get_split(method = 'scaffold', seed = 'benchmark')
# visualize label distribution
data.label_distribution()
# binarize 
data.binarize()
# convert to log
data.conver_to_log()
# get data in the various formats
data.get_data(format = 'df')
```

## Installation

```bash
pip install PyTDC
```

## TDC Dataset Overview
We have X task formulations and each is associated with many datasets. To call a dataset Y from task formulation X, simply calling ```X(name = Y)```.

### Single-instance Prediction

* <b>Absorption, Distribution, Metabolism, and Excretion</b>```ADME```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

	* <h3>Absorption</h3>

  	 Dataset Name  | Description| Reference | Type | Stats
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 Caco-2	<br> `ADME(name = 'Caco2_Wang')` </br> | The Caco-2 cell effective permeability (Peff) is an in vitro approximation of the rate at which the drug passes through intestinal tissue. | [Ning-Ning Wang, Jie Dong, Yin-Hua Deng, Min-Feng Zhu, Ming Wen, Zhi-Jiang Yao, Ai-Ping Lu, Jian-Bing Wang, and Dong-Sheng Cao. Journal of Chemical Information and Modeling 2016 56 (4), 763-773](https://pubmed.ncbi.nlm.nih.gov/27018227/) | Regression | 910 Drugs
	 HIA <br> `ADME(name = 'HIA_Hou')` </br>| The human intestinal absorption (HIA) means the process of orally administered drugs are absorbed from the gastrointestinal system into the bloodstream of the human body. | [Hou T, Wang J, Zhang W, Xu X. ADME evaluation in drug discovery. 7. Prediction of oral absorption by correlation and classification. J Chem Inf Model. 2007;47(1):208-218. doi:10.1021/ci600343x](https://pubmed.ncbi.nlm.nih.gov/17238266/) | Binary | 578 Drugs
	 Pgp <br>`ADME(name = 'Pgp_Broccatelli')` </br>| P-glycoprotein (Pgp or ABCB1) is an ABC transporter protein involved in intestinal absorption, drug metabolism, and brain penetration, and its inhibition can seriously alter a drug's bioavailability and safety. In addition, inhibitors of Pgp can be used to overcome multidrug resistance. | [A Novel Approach for Predicting P-Glycoprotein (ABCB1) Inhibition Using Molecular Interaction Fields. Fabio Broccatelli, Emanuele Carosati, Annalisa Neri, Maria Frosini, Laura Goracci, Tudor I. Oprea, and Gabriele Cruciani. Journal of Medicinal Chemistry 2011 54 (6), 1740-1751](https://pubs.acs.org/doi/abs/10.1021/jm101421d) | Binary | 1,267 Drugs
	 Bioavailability <br> `ADME(name = 'Bioavailability_Ma')` </br> | Oral bioavailability is defined as (taking the FDA's definition) “the rate and extent to which the active ingredient or active moiety is absorbed from a drug product and becomes available at the site of action”. | [Ma, Chang-Ying, et al. "Prediction models of human plasma protein binding rate and oral bioavailability derived by using GA–CG–SVM method." Journal of pharmaceutical and biomedical analysis 47.4-5 (2008): 677-682.](https://doi.org/10.1016/j.jpba.2008.03.023) | Binary | 640 Drugs
	 Bioavailability_F20_eDrug3D <br> `ADME(name = 'F20_eDrug3D')`</br> | Oral bioavailability is defined as (taking the FDA's definition) “the rate and extent to which the active ingredient or active moiety is absorbed from a drug product and becomes available at the site of action”. Processed from eDrug3D dataset. Using 20% as the threshold. | [Pihan E, Colliandre L, Guichou JF, Douguet D. e-Drug3D: 3D structure collections dedicated to drug repurposing and fragment-based drug design. Bioinformatics. 2012;28(11):1540-1541.](https://pubmed.ncbi.nlm.nih.gov/22539672/) | Binary | 403 Drugs
	 Bioavailability_F30_eDrug3D <br> `ADME(name = 'F30_eDrug3D')`</br> | Oral bioavailability is defined as (taking the FDA's definition) “the rate and extent to which the active ingredient or active moiety is absorbed from a drug product and becomes available at the site of action”. Processed from eDrug3D dataset. Using 30% as the threshold. | [Pihan E, Colliandre L, Guichou JF, Douguet D. e-Drug3D: 3D structure collections dedicated to drug repurposing and fragment-based drug design. Bioinformatics. 2012;28(11):1540-1541.](https://pubmed.ncbi.nlm.nih.gov/22539672/)| Binary | 403 Drugs

	 * <h3>Distribution</h3>

  	 Dataset Name  | Description| Reference | Type | Stats
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 BBB_Adenot <br> `ADME(name = 'BBB_Adenot')`</br> | The blood–brain barrier (BBB) is a highly selective semipermeable border of endothelial cells that prevents solutes in the circulating blood from non-selectively crossing into the extracellular fluid of the central nervous system where neurons reside. | [Adenot M, Lahana R. Blood-brain barrier permeation models: discriminating between potential CNS and non-CNS drugs including P-glycoprotein substrates. J Chem Inf Comput Sci. 2004;44(1):239-248.](https://pubmed.ncbi.nlm.nih.gov/14741033/)
	 BBB_MolNet <br>`ADME(name = 'BBB_MolNet')` </br>| The blood-brain barrier penetration (BBB) dataset is extracted from a study on the modeling and prediction of the barrier permeability. As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier blocks most drugs, hormones and neurotransmitters. Thus penetration of the barrier forms a long-standing issue in development of drugs targeting central nervous system. This dataset includes binary labels for over 2000 compounds on their permeability properties. From MoleculeNet. | [Martins, Ines Filipa, et al. "A Bayesian approach to in silico blood-brain barrier penetration modeling." Journal of chemical information and modeling 52.6 (2012): 1686-1697.](https://pubmed.ncbi.nlm.nih.gov/22612593/)
	 PPBR <br> `ADME(name = 'PPBR_Ma')` </br>| The human plasma protein binding rate (PPBR) is expressed as the percentage of a drug bound to plasma proteins. Medications attach to proteins within the blood. A drug's efficiency may be affected by the degree to which it binds. The less bound a drug is, the more efficiently it can traverse cell membranes or diffuse. | [Ma, Chang-Ying, et al. "Prediction models of human plasma protein binding rate and oral bioavailability derived by using GA–CG–SVM method." Journal of pharmaceutical and biomedical analysis 47.4-5 (2008): 677-682.](https://doi.org/10.1016/j.jpba.2008.03.023)
	 PPBR_eDrug3D <br> `ADME(name = 'PPBR_eDrug3D')` </br>| The human plasma protein binding rate (PPBR) is expressed as the percentage of a drug bound to plasma proteins. Medications attach to proteins within the blood. A drug's efficiency may be affected by the degree to which it binds. The less bound a drug is, the more efficiently it can traverse cell membranes or diffuse. Processed from eDrug3D dataset.| [Pihan E, Colliandre L, Guichou JF, Douguet D. e-Drug3D: 3D structure collections dedicated to drug repurposing and fragment-based drug design. Bioinformatics. 2012;28(11):1540-1541.](https://pubmed.ncbi.nlm.nih.gov/22539672/)
	 VD_eDrug3D <br> `ADME(name = 'VD_eDrug3D')` </br>| The volume of distribution is the theoretical volume that would be necessary to contain the total amount of an administered drug at the same concentration that it is observed in the blood plasma. Processed from eDrug3D dataset.| [Pihan E, Colliandre L, Guichou JF, Douguet D. e-Drug3D: 3D structure collections dedicated to drug repurposing and fragment-based drug design. Bioinformatics. 2012;28(11):1540-1541.](https://pubmed.ncbi.nlm.nih.gov/22539672/)


	 * <h3>Metabolism</h3>

  	 Dataset Name  | Description| Reference | Type | Stats
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 CYP2C19 <br> `ADME(name = 'CYP2C19_Veith')`</br>  | The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, the CYP P450 2C19 gene provide instructions for making an enzyme that is found primarily in liver cells in a cell structure called the endoplasmic reticulum, which is involved in protein processing and transport. | [Veith, Henrike et al. “Comprehensive characterization of cytochrome P450 isozyme selectivity across chemical libraries.” Nature biotechnology vol. 27,11 (2009): 1050-5.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2783980/); [PubChem AID1851](https://pubchem.ncbi.nlm.nih.gov/bioassay/1851) | Binary | 12,665 Drugs
	 CYP2D6 <br> `ADME(name = 'CYP2D6_Veith')`</br> | The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP2D6 is primarily expressed in the liver. It is also highly expressed in areas of the central nervous system, including the substantia nigra. | [Veith, Henrike et al. “Comprehensive characterization of cytochrome P450 isozyme selectivity across chemical libraries.” Nature biotechnology vol. 27,11 (2009): 1050-5.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2783980/); [PubChem AID1851](https://pubchem.ncbi.nlm.nih.gov/bioassay/1851)| Binary | 13,130 Drugs
	 CYP3A4 <br> `ADME(name = 'CYP3A4_Veith')`</br> | The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP3A4 is an important enzyme in the body, mainly found in the liver and in the intestine. It oxidizes small foreign organic molecules (xenobiotics), such as toxins or drugs, so that they can be removed from the body. | [Veith, Henrike et al. “Comprehensive characterization of cytochrome P450 isozyme selectivity across chemical libraries.” Nature biotechnology vol. 27,11 (2009): 1050-5.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2783980/); [PubChem AID1851](https://pubchem.ncbi.nlm.nih.gov/bioassay/1851)| Binary | 12,328 Drugs
	 CYP1A2 <br> `ADME(name = 'CYP1A2_Veith')`</br> | The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, CYP1A2 localizes to the endoplasmic reticulum and its expression is induced by some polycyclic aromatic hydrocarbons (PAHs), some of which are found in cigarette smoke. It is able to metabolize some PAHs to carcinogenic intermediates. Other xenobiotic substrates for this enzyme include caffeine, aflatoxin B1, and acetaminophen. | [Veith, Henrike et al. “Comprehensive characterization of cytochrome P450 isozyme selectivity across chemical libraries.” Nature biotechnology vol. 27,11 (2009): 1050-5.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2783980/); [PubChem AID1851](https://pubchem.ncbi.nlm.nih.gov/bioassay/1851)| Binary | 12,579 Drugs
	 CYP2C9 <br> `ADME(name = 'CYP2C9_Veith')`</br> | The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. Specifically, the CYP P450 2C9 plays a major role in the oxidation of both xenobiotic and endogenous compounds. | [Veith, Henrike et al. “Comprehensive characterization of cytochrome P450 isozyme selectivity across chemical libraries.” Nature biotechnology vol. 27,11 (2009): 1050-5.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2783980/); [PubChem AID1851](https://pubchem.ncbi.nlm.nih.gov/bioassay/1851)| Binary | 12,092 Drugs

	 * <h3>Excretion</h3>

  	 Dataset Name  | Description| Reference | Type | Stats
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 Half_life_eDrug3D <br> `ADME(name = 'HalfLife_eDrug3D')`</br> | The duration of action of a drug is known as its half life. This is the period of time required for the concentration or amount of drug in the body to be reduced by one-half. Processed from eDrug3D dataset.| [Pihan E, Colliandre L, Guichou JF, Douguet D. e-Drug3D: 3D structure collections dedicated to drug repurposing and fragment-based drug design. Bioinformatics. 2012;28(11):1540-1541. ](https://pubmed.ncbi.nlm.nih.gov/22539672/)
	 Clearance_eDrug3D <br>`ADME(name = 'Clearance_eDrug3D')` </br>| Drug clearance is concerned with the rate at which the active drug is removed from the body. Clearance is defined as the rate of drug elimination divided by the plasma concentration of the drug. Processed from eDrug3D dataset.| [Pihan E, Colliandre L, Guichou JF, Douguet D. e-Drug3D: 3D structure collections dedicated to drug repurposing and fragment-based drug design. Bioinformatics. 2012;28(11):1540-1541. ](https://pubmed.ncbi.nlm.nih.gov/22539672/)
  	 
  	 * <h3>Lipophilicity</h3>

  	 Dataset Name  | Description| Reference | Type | Stats
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 AstraZeneca <br> `ADME(name = 'Lipophilicity_AstraZeneca')`</br>  | Lipophilicity is a dataset curated from ChEMBL database containing experimental results on octanol/water distribution coefficient (logD at pH=7.4). From MoleculeNet.| [AstraZeneca. Experimental in vitro Dmpk and physicochemical data on a set of publicly disclosed compounds (2016) ](https://doi.org/10.6019/chembl3301361) | Regression | 4,200 Drugs
	 LogD74 <br> `ADME(name = 'Lipophilicity_Wang')` </br>| A high-quality hand-curated lipophilicity dataset that includes the chemical structure of 1,130 organic compounds and their n-octanol/buffer solution distribution coefficients at pH 7.4 (logD7.4). | [Wang, J-B., D-S. Cao, M-F. Zhu, Y-H. Yun, N. Xiao, Y-Z. Liang (2015). In silico evaluation of logD7.4 and comparison with other prediction methods. Journal of Chemometrics, 29(7), 389-398.](https://onlinelibrary.wiley.com/doi/full/10.1002/cem.2718) | Regression | 1,094 Drugs
	
	* <h3>Solubility</h3>

  	 Dataset Name  | Description| Reference | Type | Stats
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 AqSolDB <br> `ADME(name = 'Solubility_AqSolDB') ` </br>| AqSolDB: A curated reference set of aqueous solubility, created by the Autonomous Energy Materials Discovery [AMD] research group, consists of aqueous solubility values of 9,982 unique compounds curated from 9 different publicly available aqueous solubility datasets. | [Sorkun, M.C., Khetan, A. & Er, S. AqSolDB, a curated reference set of aqueous solubility and 2D descriptors for a diverse set of compounds. Sci Data 6, 143 (2019).](https://doi.org/10.1038/s41597-019-0151-1) | Regression | 9,982 Drugs
	 ESOL <br> `ADME(name = 'Solubility_ESOL')` </br>| ESOL (delaney) is a standard regression dataset containing structures and water solubility data for 1128 compounds. From MoleculeNet. | [Delaney, John S. "ESOL: estimating aqueous solubility directly from molecular structure." Journal of chemical information and computer sciences 44.3 (2004): 1000-1005.](https://pubs.acs.org/doi/abs/10.1021/ci034243x) | Regression | 1,128 Drugs
	 FreeSolv <br> `ADME(name = 'HydrationFreeEnergy_FreeSolv')` </br>| The Free Solvation Database, FreeSolv(SAMPL), provides experimental and calculated hydration free energy of small molecules in water. The calculated values are derived from alchemical free energy calculations using molecular dynamics simulations. From MoleculeNet.| [Mobley, David L., and J. Peter Guthrie. "FreeSolv: a database of experimental and calculated hydration free energies, with input files." Journal of computer-aided molecular design 28.7 (2014): 711-720.](https://pubmed.ncbi.nlm.nih.gov/24928188/) | Regression | 642 Drugs
	</details>

* <b>Toxicity</b>```Tox```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

  	 Dataset Name  | Description| Reference 
	 ------------  | ------------------------ | -----------
	 Tox21 <br> `Toxicity(name = 'Tox21', target = 'NR-AR')`, Choose target from [here](https://github.com/kexinhuang12345/DrugDataLoader/blob/ba2035a61897270d49af8a52d2ce51ed1571c6ee/DrugDataLoader/target_list.py#L1) </br>| 2014 Tox21 Data Challenge contains qualitative toxicity measurements for 8k compounds on 12 different targets, including nuclear receptors and stress response pathways. From MoleculeNet. | [Tox21 Challenge.](https://www.frontiersin.org/research-topics/2954/tox21-challenge-to-build-predictive-models-of-nuclear-receptor-and-stress-response-pathways-as-media)
	 ToxCast <br> `Toxicity(name = 'ToxCast', target = 'ACEA_T47D_80hr_Negative')`, Choose target from [here](https://github.com/kexinhuang12345/DrugDataLoader/blob/ba2035a61897270d49af8a52d2ce51ed1571c6ee/DrugDataLoader/target_list.py#L3) </br> | ToxCast includes qualitative results of over 600 experiments on 8k compounds. From MoleculeNet. |[Richard, Ann M., et al. "ToxCast chemical landscape: paving the road to 21st century toxicology." Chemical research in toxicology 29.8 (2016): 1225-1251.](https://pubmed.ncbi.nlm.nih.gov/27367298/)
	 ClinTox <br> `Toxicity(name = 'ClinTox')` </br>| The ClinTox dataset compares drugs that have failed clinical trials for toxicity reasons. From MoleculeNet. | [Gayvert, Kaitlyn M., Neel S. Madhukar, and Olivier Elemento. "A data-driven approach to predicting successes and failures of clinical trials." Cell chemical biology 23.10 (2016): 1294-1301.](https://pubmed.ncbi.nlm.nih.gov/27642066/)
 	</details>

* <b>High Throughput Screening</b>```HTS```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

	 Dataset Name  | Description| Reference | Type | Stats
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 SARS-CoV2 in vitro <br>`HTS(name = 'SARSCoV2_Vitro_Touret')`</br> | In-vitro screend the PRESTWICK CHEMICAL LIBRARY composed of 1,520 approved drugs in an infected cell-based assay.| [Touret, F., Gilles, M., Barral, K. et al. In vitro screening of a FDA approved chemical library reveals potential inhibitors of SARS-CoV-2 replication. Sci Rep 10, 13093 (2020).](https://doi.org/10.1038/s41598-020-70143-6) | Binary | 
	 SARS-CoV2 3CLPro <br>`HTS(name = 'SARSCoV2_3CLPro_Diamond')`</br> | A large XChem crystallographic fragment screen against SARS-CoV-2 main protease at high resolution. | [Diamond Light Source](https://www.diamond.ac.uk/covid-19/for-scientists/Main-protease-structure-and-XChem.html) | Binary | 
	 HIV <br>`HTS(name = 'HIV')`</br> | The HIV dataset was introduced by the Drug Therapeutics Program (DTP) AIDS Antiviral Screen, which tested the ability to inhibit HIV replication for over 40,000 compounds. From MoleculeNet. | [AIDS Antiviral Screen Data. https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data](placeholder) | Binary | 41,127 Drugs
	</details>

* <b>Quantum Mechanics</b>```QM```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

  	Dataset Name  | Description| Reference | Type | Stats
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	QM7 <br> `QM(name = 'QM7, target = 'X')` Choose target from [here](https://github.com/kexinhuang12345/DrugDataLoader/blob/master/DrugDataLoader/target_list.py#L294) </br> | This dataset is for multitask learning where 14 properties (e.g. polarizability, HOMO and LUMO eigenvalues, excitation energies) have to be predicted at different levels of theory (ZINDO, SCS, PBE0, GW). From MoleculeNet.| [ML. Ruddigkeit, R. van Deursen, L. C. Blum, J.-L. Reymond, Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17, J. Chem. Inf. Model. 52, 2864–2875, 2012.](http://dx.doi.org/10.1021/ja00051a040) | Regression | 7,211 drugs
	QM8 <br> `QM(name = 'QM8, target = 'X')` Choose target from [here](https://github.com/kexinhuang12345/DrugDataLoader/blob/master/DrugDataLoader/target_list.py#L296) </br> | TElectronic spectra and excited state energy of small molecules calculated by multiple quantum mechanic methods. From MoleculeNet.|  [ML. Ruddigkeit, R. van Deursen, L. C. Blum, J.-L. Reymond, Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17, J. Chem. Inf. Model. 52, 2864–2875, 2012.](http://dx.doi.org/10.1021/ja00051a040) | Regression | 22,000 drugs 
	QM9 <br> `QM(name = 'QM9, target = 'X')` Choose target from [here](https://github.com/kexinhuang12345/DrugDataLoader/blob/master/DrugDataLoader/target_list.py#L298) </br> | Geometric,  energetic, electronic and thermodynamic properties of DFT-modelled small molecules. From MoleculeNet.|  [R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld, Quantum chemistry structures and properties of 134 kilo molecules, Scientific Data 1, 140022, 2014.](http://www.nature.com/articles/sdata201422) | Regression | 22,000 drugs 
	</details>

* <b>Paratope Prediction</b>```Paratope```
	<details>
	<summary>CLICK HERE FOR THE DATASETS!</summary>

	Dataset Name  | Description| Reference | Type | Stats
	------------  | ------------------------ | ----------- | ----------- | -----------
	</details>

* <b>Epitope Prediction</b>```Epitope```
	<details>
	<summary>CLICK HERE FOR THE DATASETS!</summary>

	Dataset Name  | Description| Reference | Type | Stats
	------------  | ------------------------ | ----------- | ----------- | -----------
	Bepipred_IEDB <br> `Epitope(name = 'Bepipred_IEDB)` </br> | | []() | Token Classification | X antigens
	Bepipred_PDB <br> `Epitope(name = 'Bepipred_PDB)` </br> | | []() | Token Classification | X antigens
	</details>

* <b>Developability</b>```Develop```
	<details>
	<summary>CLICK HERE FOR THE DATASETS!</summary>

	Dataset Name  | Description| Reference | Type | Stats
	------------  | ------------------------ | ----------- | ----------- | -----------
	CDR_TAP <br> `Develop(name = 'CDR_TAP)` </br> | | []() | Regression | X antibodies
	</details>

* <b>Reaction Yields</b>```Yields```
	<details>
	<summary>CLICK HERE FOR THE DATASETS!</summary>

	Dataset Name  | Description| Reference | Type | Stats
	------------  | ------------------------ | ----------- | ----------- | -----------
	</details>

### Multi-instance Prediction

* <b>Drug-Target Interaction</b>```DTI```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

	 Dataset Name  | Description| Reference | Type | Stats (pairs/#drugs/#targets)
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 BindingDB <br> `DTI(name = 'BindingDB_X')` Choose X from Kd, IC50, EC50, or Ki </br> | BindingDB is a public, web-accessible database of measured binding affinities, focusing chiefly on the interactions of protein considered to be drug-targets with small, drug-like molecules. | [BindingDB: a web-accessible database of experimentally determined protein–ligand binding affinities](https://academic.oup.com/nar/article-abstract/35/suppl_1/D198/1119109) | Regression (log)/Binary | 66,444/10,665/1,413 for Kd, 1,073,803/549,205/5,078 for IC50, 151,413/91,773/1,240 for EC50, 41,0478/174,662/3,070 for Ki
	 DAVIS  <br> `DTI(name = 'DAVIS')` </br> | The interaction of 72 kinase inhibitors with 442 kinases covering >80% of the human catalytic protein kinome. | [Davis, M., Hunt, J., Herrgard, S. et al. Comprehensive analysis of kinase inhibitor selectivity. Nat Biotechnol 29, 1046–1051 (2011).](https://www.nature.com/articles/nbt.1990) | Regression (log)/Binary | 30,056/68/379
	 KIBA  <br> `DTI(name = 'KIBA')`  </br>| An integrated drug-target bioactivity matrix across 52,498 chemical compounds and 467 kinase targets, including a total of 246,088 KIBA scores, has been made freely available. | [Tang J, Szwajda A, Shakyawar S, et al. Making sense of large-scale kinase inhibitor bioactivity data sets: a comparative and integrative analysis. J Chem Inf Model. 2014;54(3):735-743.](https://pubmed.ncbi.nlm.nih.gov/24521231/) | Regression | 118,254/2,068/229
	</details>

* <b>Drug-Drug Interaction</b>```DDI```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

  	 Dataset Name  | Description| Reference  | Type | Stats (pairs/#drugs)
	 ------------ | ------------------------ | ----------- | ----------- | -----------
	 DrugBank | DrugBank drug-drug interaction dataset is manually sourced from FDA/Health Canada drug labels as well as primary literature. It has 86 interaction types. Drug SMILES is provided. | [Wishart DS, et al. (2017) DrugBank 5.0: A major update to the DrugBank database for 2018. Nucleic Acids Res 46:D1074–D1082.](https://academic.oup.com/nar/article/46/D1/D1074/4602867) | Multi-Class/Network | 191,519/1,706
	 TWOSIDES | Polypharmacy side-effects are associated with drug pairs (or higher-order drug combinations) and cannot be attributed to either individual drug in the pair (in a drug combination). | [Tatonetti, Nicholas P., et al. Data-driven prediction of drug effects and interactions. Science Translational Medicine. 2012.](https://stm.sciencemag.org/content/4/125/125ra31.short) | Multi-Label/Network | 4,649,441/645
	</details>

* <b>Protein-Protein Interaction</b>```PPI```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

	 Dataset Name  | Description| Reference | Type | Stats (pairs/#proteins)
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 HuRI <br> `PPI(name = 'HuRI)` </br> | All pairwise combinations of human protein-coding genes are systematically being interrogated to identify which are involved in binary protein-protein interactions. In our most recent effort 17,500 proteins have been tested and a first human reference interactome (HuRI) map has been generated. From the Center for Cancer Systems Biology at Dana-Farber Cancer Institute. Note that the feature is peptide sequence, if a protein gene is associated with multiple peptides, we separate them by '\*'.| [Luck, K., Kim, D., Lambourne, L. et al. A reference map of the human binary protein interactome. Nature 580, 402–408 (2020). ](https://doi.org/10.1038/s41586-020-2188-x) | Binary/Network | 51,813/8,248
	</details>

* <b>Gene-Disease Interaction</b>```GDI```
	<details>
	<summary>CLICK HERE FOR THE DATASETS!</summary>

	Dataset Name  | Description| Reference | Type | Stats
	------------  | ------------------------ | ----------- | ----------- | -----------
	</details>

* <b>MicroRNA-Target Interaction</b>```MTI```
	<details>
	<summary>CLICK HERE FOR THE DATASETS!</summary>

	Dataset Name  | Description| Reference | Type | Stats
	------------  | ------------------------ | ----------- | ----------- | -----------
	</details>

* <b>Drug Response Prediction</b>(Cell lines-Monotherapy Affinity)```DrugResponse```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

	 Dataset Name  | Description| Reference | Type | Stats (#pairs/#cell lines/#drugs)
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 GDSC1 <br> `DrugResponse(name = 'GDSC1)` </br> | Genomics in Drug Sensitivity in Cancer (GDSC) is a resource for therapeutic biomarker discovery in cancer cells. It contains wet lab IC50 for 100s of drugs in 1000 cancer cell lines. In this dataset, we use RMD normalized gene expression for cancer lines and SMILES for drugs. Y is the log normalized IC50. This is the version 1 of GDSC. | [Yang, Wanjuan, et al. "Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells." Nucleic acids research 41.D1 (2012): D955-D961.](https://academic.oup.com/nar/article-abstract/41/D1/D955/1059448) | Regression | 177,310/958/208
	 GDSC2 <br> `DrugResponse(name = 'GDSC2)` </br> | Genomics in Drug Sensitivity in Cancer (GDSC) is a resource for therapeutic biomarker discovery in cancer cells. It contains wet lab IC50 for 100s of drugs in 1000 cancer cell lines. In this dataset, we use RMD normalized gene expression for cancer lines and SMILES for drugs. Y is the log normalized IC50. This is the version 2 of GDSC, which uses improved experimental procedures. | [Yang, Wanjuan, et al. "Genomics of Drug Sensitivity in Cancer (GDSC): a resource for therapeutic biomarker discovery in cancer cells." Nucleic acids research 41.D1 (2012): D955-D961.](https://academic.oup.com/nar/article-abstract/41/D1/D955/1059448) | Regression | 92,703/805/137
	</details>

* <b>Drug Synergy Prediction</b>(Cell lines-Combotherapy Affinity)```DrugSyn```
	<details>
	<summary>CLICK HERE FOR THE DATASETS!</summary>

	Dataset Name  | Description| Reference | Type | Stats
	------------  | ------------------------ | ----------- | ----------- | -----------
	</details>

* <b>Peptide-MHC Binding Prediction</b>```PeptideMHC```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

	 Dataset Name  | Description| Reference | Type | Stats (pairs/#peptides/#ofMHCs)
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 MHC1_NetMHCpan <br> `PeptideMHC(name = 'MHC1_NetMHCpan')` </br> | Binding of peptides to MHC class I molecules (MHC-I) is essential for antigen presentation to cytotoxic T-cells. An organized datasets for MHC class I collected from IEDB and IMGT/HLA database. | [Nielsen, Morten, and Massimo Andreatta. "NetMHCpan-3.0; improved prediction of binding to MHC class I molecules integrating information from multiple receptor and peptide length datasets." Genome medicine 8.1 (2016): 1-9.](https://genomemedicine.biomedcentral.com/articles/10.1186/s13073-016-0288-x) | Regression | 185,985/43,018/150
	 MHC2_NetMHCIIpan <br> `PeptideMHC(name = 'MHC2_NetMHCIIpan')` </br> | Major histocompatibility complex class II (MHC‐II) molecules are found on the surface of antigen‐presenting cells where they present peptides derived from extracellular proteins to T helper cells. Useful to identify T‐cell epitopes. An organized datasets for MHC class II collected from IEDB database. | [Jensen, Kamilla Kjaergaard, et al. "Improved methods for predicting peptide binding affinity to MHC class II molecules." Immunology 154.3 (2018): 394-406.](https://onlinelibrary.wiley.com/doi/full/10.1111/imm.12889) | Regression | 134,281/17,003/75
	</details>

* <b>Antibody-Antigen Affinity Prediction</b>```AntibodyAff```
	<details>
	<summary>CLICK HERE FOR THE DATASETS!</summary>

	Dataset Name  | Description| Reference | Type | Stats
	------------  | ------------------------ | ----------- | ----------- | -----------
	ProteinAntigen_SAbDab <br> `AntibodyAff(name = 'ProteinAntigen_SAbDab')` </br> | Antibody-antigen affinity dataset. Processed from SAbDab dataset, only uses protein/peptide antigens for sequence compatbility. The features are amino acid sequence.| [Dunbar, James, et al. "SAbDab: the structural antibody database." Nucleic acids research 42.D1 (2014): D1140-D1146.](https://academic.oup.com/nar/article-abstract/42/D1/D1140/1044118) | Regression | 493
	</details>

* <b>Catalyst Prediction</b>```Catalyst```
	<details>
	<summary>CLICK HERE FOR THE DATASETS!</summary>

	Dataset Name  | Description| Reference | Type | Stats
	------------  | ------------------------ | ----------- | ----------- | -----------
	</details>

### Generation

* <b>Distribution Molecule Generation</b>```DistMolGen```
	<details>
	<summary>CLICK HERE FOR THE DATASETS!</summary>

	Dataset Name  | Description| Reference | Type | Stats
	------------  | ------------------------ | ----------- | ----------- | -----------
	</details>

* <b>Goal-oriented Molecule Generation</b>```GoalMolGen```
	<details>
	<summary>CLICK HERE FOR THE DATASETS!</summary>

	Dataset Name  | Description| Reference | Type | Stats
	------------  | ------------------------ | ----------- | ----------- | -----------
	</details>

* <b>Paired Molecule Generation</b>```PairMolGen```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

	 Dataset Name  | Description| Reference | Type | Stats (#pairs/#drugs)
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 DRD2 <br> `MolGenPaired(name = 'DRD2')`</br> |  | | | 34,404/21,703 
	 QED <br> `MolGenPaired(name = 'QED')`</br>|||| 88,306/52,262 
	 logP <br> `MolGenPaired(name = 'LogP')`</br>|||| 99,909/99,794 
	 JNK3 ||||
	 GSK-3beta ||||
	</details>	

* <b>Retrosynthesis</b>```RetroSyn```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

	 Dataset Name  | Description| Reference | Type | Stats (#drugs)
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 USPTO-50K | | | |
	</details>	

* <b>Reaction Forward Prediction</b>```Reaction```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

	 Dataset Name  | Description| Reference | Type | Stats (#drugs)
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	 USPTO-50K | | | |
	</details>	


## Data Split

To retrieve the dataset split, you could simply type
```python 
data = X(name = Y)
data.get_split(seed = 'benchmark')
# {'train': df_train, 'val': df_val, ''test': df_test}
```
You can specify the splitting method, random seed, and split fractions in the function by e.g. `data.get_split(method = 'cold_drug', seed = 1, frac = [0.7, 0.1, 0.2])`. For drug property prediction, a scaffold split function is also provided. Simply set `method = 'scaffold'`. 

## Benchmark and Leaderboard

We are actively working on the benchmark and leaderboard methods. We would release this feature in the next major release. In the meantime, if you have expertise or interest in helping build this feature, please send emails to kexinhuang@hsph.harvard.edu.

## Contribute

TDC is designed to be a community-driven effort. If you have new dataset or task that wants to be included in TDC, please reachout to kexinhuang@hsph.harvard.edu. 

## Contact

Send emails to kexinhuang@hsph.harvard.edu or open an issue.

## Disclaimer

TDC is an open-source effort. Many datasets are aggregated from various public website sources. We use the Attribution-NonCommercial-ShareAlike 4.0 International license to suffice many datasets requirement. If it still infringes the copyright of the dataset author, please let us know and we will take it down ASAP.

