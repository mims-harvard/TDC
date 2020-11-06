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

```bash
python setup.py install 
### developer installation
```

## TDC Dataset Overview
We have X task formulations and each is associated with many datasets. To call a dataset Y from task formulation X, simply calling ```X(name = Y)```.

### Single-instance Prediction

* <b>Quantum Mechanics</b>```QM```
	<details>
  	<summary>CLICK HERE FOR THE DATASETS!</summary>

  	Dataset Name  | Description| Reference | Type | Stats
	 ------------  | ------------------------ | ----------- | ----------- | -----------
	QM7 <br> `QM(name = 'QM7, target = 'X')` Choose target from [here](https://github.com/kexinhuang12345/DrugDataLoader/blob/master/DrugDataLoader/target_list.py#L294) </br> | This dataset is for multitask learning where 14 properties (e.g. polarizability, HOMO and LUMO eigenvalues, excitation energies) have to be predicted at different levels of theory (ZINDO, SCS, PBE0, GW). From MoleculeNet.| [ML. Ruddigkeit, R. van Deursen, L. C. Blum, J.-L. Reymond, Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17, J. Chem. Inf. Model. 52, 2864–2875, 2012.](http://dx.doi.org/10.1021/ja00051a040) | Regression | 7,211 drugs
	QM8 <br> `QM(name = 'QM8, target = 'X')` Choose target from [here](https://github.com/kexinhuang12345/DrugDataLoader/blob/master/DrugDataLoader/target_list.py#L296) </br> | TElectronic spectra and excited state energy of small molecules calculated by multiple quantum mechanic methods. From MoleculeNet.|  [ML. Ruddigkeit, R. van Deursen, L. C. Blum, J.-L. Reymond, Enumeration of 166 billion organic small molecules in the chemical universe database GDB-17, J. Chem. Inf. Model. 52, 2864–2875, 2012.](http://dx.doi.org/10.1021/ja00051a040) | Regression | 22,000 drugs 
	QM9 <br> `QM(name = 'QM9, target = 'X')` Choose target from [here](https://github.com/kexinhuang12345/DrugDataLoader/blob/master/DrugDataLoader/target_list.py#L298) </br> | Geometric,  energetic, electronic and thermodynamic properties of DFT-modelled small molecules. From MoleculeNet.|  [R. Ramakrishnan, P. O. Dral, M. Rupp, O. A. von Lilienfeld, Quantum chemistry structures and properties of 134 kilo molecules, Scientific Data 1, 140022, 2014.](http://www.nature.com/articles/sdata201422) | Regression | 22,000 drugs 
	</details>


### Multi-instance Prediction


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
	SAbDab <br> `AntibodyAff(name = 'ProteinAntigen_SAbDab')` </br> | Antibody-antigen affinity measures the efficacy of the antibody to the antigen. Processed from SAbDab dataset, where we only use protein/peptide antigens for sequence compatbility. The features are amino acid sequence.| [Dunbar, James, et al. "SAbDab: the structural antibody database." Nucleic acids research 42.D1 (2014): D1140-D1146.](https://academic.oup.com/nar/article-abstract/42/D1/D1140/1044118) | Regression | 493
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

