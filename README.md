<p align="center"><img src="https://raw.githubusercontent.com/mims-harvard/TDC/master/fig/logo.png" alt="logo" width="600px" /></p>

----

[![Website](https://img.shields.io/badge/Homepage-docs-<COLOR>.svg)](https://zitniklab.hms.harvard.edu/TDC/)
[![PyPI version](https://badge.fury.io/py/PyTDC.svg)](https://badge.fury.io/py/PyTDC)
[![Downloads](https://pepy.tech/badge/pytdc)](https://pepy.tech/project/pytdc)
[![GitHub Repo stars](https://img.shields.io/github/stars/mims-harvard/TDC)](https://github.com/mims-harvard/TDC/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/mims-harvard/TDC)](https://github.com/mims-harvard/TDC/network/members)
[![Build Status](https://travis-ci.org/mims-harvard/TDC.svg?branch=master)](https://travis-ci.org/mims-harvard/TDC)
[![TDC CircleCI](https://circleci.com/gh/mims-harvard/TDC.svg?style=svg)](https://app.circleci.com/pipelines/github/mims-harvard/TDC)

This repository hosts **Therapeutics Data Commons (TDC)**, an open, user-friendly and extensive machine learning dataset hub for therapeutics. So far, it includes more than 50+ datasets for 20+ tasks (ranging from target identification, virtual screening, QSAR to manufacturing, safety surveillance and etc) in many therapeutics development stages across small molecules and biologics. 

[**Project Website**](https://zitniklab.hms.harvard.edu/TDC/)

[**Join TDC Mailing List**](https://groups.io/g/tdc)

**Invited talk at the [National Symposium on Drug Repurposing for Future Pandemics (#futuretx20)](https://www.drugsymposium.org/)** [**\[Slides\]**](https://drive.google.com/file/d/11eTrh_lsqPcwu3RZRYjJGNpJ3s18YlBS/view)


## Updates
- `0.1.2`: The first TDC Leaderboard is released! Checkout the leaderboard guide [here](https://zitniklab.hms.harvard.edu/TDC/benchmark/overview/) and the ADMET Leaderboard [here](https://zitniklab.hms.harvard.edu/TDC/benchmark/admet_group/).
- `0.1.1`: Replaced VD, Half Life and Clearance datasets from new sources that have higher qualities. Added LD50 to Tox.
- `0.1.0`: Molecule quality check for ADME, Toxicity and HTS (canonicalized, and remove error mols).
- `0.0.9`: Added DrugComb NCI-60, CYP2C9/2D6/3A4 substrates, Carcinogens toxicity! 
- `0.0.8`: Added hREG, DILI, Skin Reaction, Ames Mutagenicity, PPBR from AstraZeneca; added meta oracles!

## Features

- *From Bench to Bedside*: covers 70+ datasets for 20+ tasks in numerous therapeutics development stages across small molecules and biologics.
- *User-friendly*: 3 lines of codes to access any dataset and hassle-free installation.
- *Ready-to-use*: the dataset is processed into machine learning ready format. 
- *Data functions*: TDC supports various useful functions such as data evaluators, realistic data split functions, data processing helpers, and molecule generation oracles! 
- *Leaderboard*: provides a benchmark for fair model comparison across many therapeutics tasks.
- *Community-driven effort*: TDC is a community-driven effort. Contact us if you want to contribute a new dataset or task!

<p align="center"><img src="https://raw.githubusercontent.com/mims-harvard/TDC/master/fig/tdc_overview.png" alt="overview" width="600px" /></p>

## Installation

### Using `pip`

To install the core environment dependencies of TDC, use `pip`:

```bash
pip install PyTDC
```

**Note**: TDC is in beta release. Please update your local copy regularly by

```bash
pip install PyTDC --upgrade
```

The core data loaders are designed to be lightweight, thus has minimum package dependency:

```bash
numpy, pandas, tqdm, scikit-learn, fuzzywuzzy
```

For other utilities requiring extra dependencies, TDC will print out the relevant installation instruction. To install the full dependencies, please consider use the below conda-forge solution. 

### Using `conda`

To use data functions such as molecule oracles, scaffold split, etc., they require packages such as RDKit. To do that, use the below `conda` installation: 

```bash
conda install -c conda-forge pytdc
```


## Cite Us

If you found our work useful, please cite us:
```
@misc{tdc,
  author={Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Zitnik, Marinka},
  title={Therapeutics Data Commons: Machine Learning Datasets for Therapeutics},
  howpublished={\url{http://tdc.hms.harvard.edu}},
  month=nov,
  year=2020
}
```
Paper is in progress and will come out soon.


## TDC Data Loader

TDC covers a wide range of therapeutics tasks with varying data structures. Thus, we organize it into three layers of hierarchies. First, we divide into three distinctive machine learning **problems**:

* Single-instance prediction `single_pred`: Prediction of property given individual biomedical entity.
* Multi-instance prediction `multi_pred`: Prediction of property given multiple biomedical entities. 
* Generation `generation`: Generation of new biomedical entity.

The second layer is **task**. Each therapeutic task falls into one of the machine learning problem. We create a data loader class for every *task* that inherits from the base problem data loader. 

The last layer is **dataset**, where each task consists of many of them. As the data structure of most datasets in a task is the same, the dataset is used as a function input to the task data loader.

Supposed a dataset X is from therapeutic task Y with machine learning problem Z, then to obtain the data and splits, simply type:

```python
from tdc.Z import Y
data = Y(name = 'X')
splits = data.split()
```
For example, to obtain the HIA dataset from ADME therapeutic task in the single-instance prediction problem:

```python
from tdc.single_pred import ADME
data = ADME(name = 'HIA_Hou')
# split into train/val/test using benchmark seed and split methods
split = data.get_split(method = 'scaffold', seed = 'benchmark')
# get the entire data in the various formats
data.get_data(format = 'df')
```

You can see all the datasets belonging to a task via:
```python
from tdc.utils import retrieve_dataset_names
retrieve_dataset_names('ADME')
```

Explore all therapeutic tasks and datasets in the [website](https://zitniklab.hms.harvard.edu/TDC/overview/)!

## TDC Data Functions

#### Data Split

To retrieve the training/validation/test dataset split, you could simply type
```python 
data = X(name = Y)
data.get_split(seed = 'benchmark')
# {'train': df_train, 'val': df_val, ''test': df_test}
```
You can specify the splitting method, random seed, and split fractions in the function by e.g. `data.get_split(method = 'scaffold', seed = 1, frac = [0.7, 0.1, 0.2])`. Check out the [data split page](https://zitniklab.hms.harvard.edu/TDC/functions/data_split/) on the website for details.

#### Model Evaluation

We provide various evaluation metrics for the tasks in TDC, which are described in [model evaluation page](https://zitniklab.hms.harvard.edu/TDC/functions/data_evaluation/) on the website. For example, to use metric ROC-AUC, you could simply type

```python
from tdc import Evaluator
evaluator = Evaluator(name = 'ROC-AUC')
score = evaluator(y_true, y_pred)
```

#### Data Processing 

We provide numerous data processing helper functions such as label transformation, data balancing, pair data to PyG/DGL graphs, negative sampling, database querying and so on. For individual function usage, please checkout the [data processing page](https://zitniklab.hms.harvard.edu/TDC/functions/data_process/) on the website.

#### Molecule Generation Oracles

For molecule generation tasks, we provide 10+ oracles for both goal-oriented and distribution learning. For detailed usage of each oracle, please checkout the [oracle page](https://zitniklab.hms.harvard.edu/TDC/functions/oracles/) on the website. For example, we want to retrieve the GSK3Beta oracle:

```python
from tdc import Oracle
oracle = Oracle(name = 'GSK3B')
oracle(['CC(C)(C)....' 
	'C[C@@H]1....',
	'CCNC(=O)....', 
	'C[C@@H]1....'])

# [0.03, 0.02, 0.0, 0.1]
```
Note that the graph-to-graph paired molecule generation is provided as separate [datasets](https://zitniklab.hms.harvard.edu/TDC/generation_tasks/pairmolgen/). 


## TDC Leaderboard

TDC hosts a series of leaderboards for researchers to keep abreast with the state-of-the-art models on therapeutics tasks.

Each dataset in TDC is a benchmark. But for a machine learning model to be useful for a specific downstream therapeutic usage, the model has to achieve consistently good performance across a set of datasets or tasks. Motivated by this, TDC intentionally group individual benchmarks into a benchmark group. Datasets in a benchmark group are centered around a theme and are all carefully selected. The dataset split and evaluation metrics are also carefully selected to reflect real-world challenges.

TDC provides a programming framework to access the data in a benchmark group. We use ADMET group as an example.

```python
from tdc import BenchmarkGroup
group = BenchmarkGroup(name = 'ADMET_Group', path = 'data/')
predictions = {}

for benchmark in group:
    name = benchmark['name']
    train, valid, test = benchmark['train'], benchmark['valid'], benchmark['test']
    ## --- train your model --- ##
    predictions[name] = y_pred

group.evaluate(predictions)
# {'caco2_wang': {'mae': 0.234}, 'hia_hou': {'roc-auc': 0.786}, ...}
```

For more functions of the `BenchmarkGroup` class, please visit [here](https://zitniklab.hms.harvard.edu/TDC/benchmark/overview/).

## Tutorials

We provide a series of tutorials for you to get started using TDC:

| Name  | Description                                             |
|-------|---------------------------------------------------------|
| [101](tutorials/TDC_101_Data_Loader.ipynb)   | Introduce TDC Data Loaders                              |
| [102](tutorials/TDC_102_Data_Functions.ipynb)   | Introduce TDC Data Functions                            |
| [103.1](tutorials/TDC_103.1_Datasets_Small_Molecules.ipynb) | Walk through TDC Small Molecule Datasets                |
| [103.2](tutorials/TDC_103.2_Datasets_Biologics.ipynb) | Walk through TDC Biologics Datasets                     |
| [104](tutorials/TDC_104_ML_Model_DeepPurpose.ipynb)   | Generate 21 ADME ML Predictors with 15 Lines of Code |
| [105](tutorials/TDC_105_Oracle.ipynb)   | Molecule Generation Oracles                             |


## Contribute

TDC is designed to be a community-driven effort. If you have new dataset or task or data function that wants to be included in TDC, please fill in this [form](https://kexinhuang.typeform.com/to/W5DKjXDg)!

## Contact

Send emails to [us](mailto:kexinhuang@hsph.harvard.edu) or open an issue.

## Data Server Maintenance Issues

TDC is hosted in [Harvard Dataverse](https://dataverse.harvard.edu/). When dataverse is under maintenance, TDC will not able to retrieve datasets. Although rare, when it happens, please come back in couple of hours or check the status by visiting the [dataverse website](https://dataverse.harvard.edu/).

## License
TDC codebase is under MIT license. For individual dataset usage, please refer to the dataset license in the website.
