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

* <b>Antibody-Antigen Affinity Prediction</b>```AntibodyAff```
	<details>
	<summary>CLICK HERE FOR THE DATASETS!</summary>

	Dataset Name  | Description| Reference | Type | Stats
	------------  | ------------------------ | ----------- | ----------- | -----------
	SAbDab <br> `AntibodyAff(name = 'ProteinAntigen_SAbDab')` </br> | Antibody-antigen affinity measures the efficacy of the antibody to the antigen. Processed from SAbDab dataset, where we only use protein/peptide antigens for sequence compatbility. The features are amino acid sequence.| [Dunbar, James, et al. "SAbDab: the structural antibody database." Nucleic acids research 42.D1 (2014): D1140-D1146.](https://academic.oup.com/nar/article-abstract/42/D1/D1140/1044118) | Regression | 493
	</details>

## Cite Us

If you found our work useful, please consider cite us:
```
@misc{tdc,
  author       = {Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Zitnik, Marinka},
  title        = {Therapeutics Data Commons: Machine Learning Datasets for Therapeutics},
  howpublished = {\url{http://tdc.hms.harvard.edu}},
  month        = nov,
  year         = 2020
}
```
Paper will also be released soon.

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

