<p align="center"><img src="https://raw.githubusercontent.com/mims-harvard/TDC/master/fig/logo.png" alt="logo" width="600px" /></p>

----

[![website](https://img.shields.io/badge/website-live-brightgreen)](https://zitniklab.hms.harvard.edu/TDC/)
[![PyPI version](https://badge.fury.io/py/PyTDC.svg)](https://badge.fury.io/py/PyTDC)
[![Downloads](https://pepy.tech/badge/pytdc)](https://pepy.tech/project/pytdc)
[![GitHub Repo stars](https://img.shields.io/github/stars/mims-harvard/TDC)](https://github.com/mims-harvard/TDC/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/mims-harvard/TDC)](https://github.com/mims-harvard/TDC/network/members)
[![Build Status](https://travis-ci.org/mims-harvard/TDC.svg?branch=master)](https://travis-ci.org/mims-harvard/TDC)
[![TDC CircleCI](https://circleci.com/gh/mims-harvard/TDC.svg?style=svg)](https://app.circleci.com/pipelines/github/mims-harvard/TDC)

[**Project Website**](https://zitniklab.hms.harvard.edu/TDC/) | [**TDC Mailing List**](https://groups.io/g/tdc) | [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40ProjectTDC)](https://twitter.com/ProjectTDC)

**Therapeutics Data Commons (TDC)** is a collection of machine learning tasks spread across different domains of therapeutics.

Therapeutics machine learning is an exciting field with incredible opportunities for expansion, innovation, and impact. Datasets and benchmarks in TDC provide a systematic model development and evaluation framework that allows more machine learning researchers to contribute to the field.

We envision that TDC can considerably accelerate machine-learning model development, validation and transition into production and clinical implementation. 

**Invited talk at the [Harvard Symposium on Drugs for Future Pandemics (#futuretx20)](https://www.drugsymposium.org/)** [**\[Slides\]**](https://drive.google.com/file/d/11eTrh_lsqPcwu3RZRYjJGNpJ3s18YlBS/view) [**\[Video\]**](https://youtu.be/ZuCOhEZtaOw)


## Updates
- `0.1.5`: Added four realistic oracles from docking scores and synthetic accessibility! Checkout [here](https://zitniklab.hms.harvard.edu/TDC/functions/oracles/)!
- `0.1.4`: Added the 1st version of [`MolConvert`](https://zitniklab.hms.harvard.edu/TDC/functions/data_process/#molecule-conversion) class that can map among ~15 molecular formats in 2 lines of code (For 2D: from SMILES/SEFLIES and convert to SELFIES/SMILES, Graph2D, PyG, DGL, ECFP2-6, MACCS, Daylight, RDKit2D, Morgan, PubChem; For 3D: from XYZ, SDF files to Graph3D, Columb Matrix); Also a quality check on DTI datasets with IDs added.
- Checkout **[Contribution Guide](CONTRIBUTE.md)** to add new dataset, task, function!
- `0.1.3`: Added new therapeutics task on CRISPR Repair Outcome Prediction! Added a data function to map molecule to popular cheminformatics fingerprint.
- `0.1.2`: The first TDC Leaderboard is released! Checkout the leaderboard guide [here](https://zitniklab.hms.harvard.edu/TDC/benchmark/overview/) and the ADMET Leaderboard [here](https://zitniklab.hms.harvard.edu/TDC/benchmark/admet_group/).
- `0.1.1`: Replaced VD, Half Life and Clearance datasets from new sources that have higher qualities. Added LD50 to Tox.
- `0.1.0`: Molecule quality check for ADME, Toxicity and HTS (canonicalized, and remove error mols).
- `0.0.9`: Added DrugComb NCI-60, CYP2C9/2D6/3A4 substrates, Carcinogens toxicity! 
- `0.0.8`: Added hREG, DILI, Skin Reaction, Ames Mutagenicity, PPBR from AstraZeneca; added meta oracles!

## Features

- *Diverse areas of therapeutics development*: TDC covers a wide range of learning tasks, including target discovery, activity screening, efficacy, safety, and manufacturing across biomedical products, including small molecules, antibodies, and vaccines.
- *Ready-to-use datasets*: TDC is minimally dependent on external packages. Any TDC dataset can be retrieved using only 3 lines of code.
- *Data functions*: TDC provides extensive data functions, including data evaluators, meaningful data splits, data processors, and molecule generation oracles. 
- *Leaderboards*: TDC provides benchmarks for fair model comparison and a systematic model development and evaluation.
- *Open-source initiative*: TDC is an open-source initiative. If you want to get involved, let us know. 

<p align="center"><img src="https://raw.githubusercontent.com/mims-harvard/TDC/master/fig/tdc_overview.png" alt="overview" width="600px" /></p>

## Installation

### Using `pip`

To install the core environment dependencies of TDC, use `pip`:

```bash
pip install PyTDC
```

**Note**: TDC is in the beta release. Please update your local copy regularly by

```bash
pip install PyTDC --upgrade
```

The core data loaders are lightweight with minimum dependency on external packages:

```bash
numpy, pandas, tqdm, scikit-learn, fuzzywuzzy
```

For utilities requiring extra dependencies, TDC prints installation instructions. To install full dependencies, please use the following `conda-forge` solution. 

### Using `conda`

Data functions for molecule oracles, scaffold split, etc., require certain packages like RDKit. To install those packages, use the following `conda` installation: 

```bash
conda install -c conda-forge pytdc
```

## Cite Us

If you found our work useful, please cite us:

```
@misc{tdc,
  author={Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Roohani, Yusuf and Leskovec, Jure and Coley, Connor and Xiao, Cao and Sun, Jimeng and Zitnik, Marinka},
  title={Therapeutics Data Commons: Machine Learning Datasets for Therapeutics},
  howpublished={\url{https://zitniklab.hms.harvard.edu/TDC/}},
  month=nov,
  year=2020
}
```

We are now preparing a manuscript, which we will release soon.

## Design of TDC

TDC has a unique three-layer hierarchical structure, which to our knowledge is the first attempt at systematically evaluating machine learning across the field of therapeutics. We organize TDC into three distinct *problems*, each of these problems is instantiated through a variety of *learning tasks*, and each task, in turn, is evaluated on a series of *datasets*.

We choose our datasets to highlight three major areas (*problems*) where machine learning can facilitate scientific advances: single-instance prediction, multi-instance prediction, and generative modeling:

* Single-instance prediction `single_pred`: Prediction of property given individual biomedical entity.
* Multi-instance prediction `multi_pred`: Prediction of property given multiple biomedical entities. 
* Generation `generation`: Generation of a new biomedical entity.

<p align="center"><img src="https://raw.githubusercontent.com/mims-harvard/TDC/master/fig/tdc_problems.png" alt="problems" width="500px" /></p>

The second layer in the TDC structure is organized into *learning tasks*. Improvement on these tasks range in applications, including designing new antibodies, identifying personalized combinatorial therapies, improving disease diagnosis, and finding new cures for emerging diseases. 

Finally, in the third layer of TDC, each task is instantiated via multiple *datasets*. For each dataset, we construct data splits to simulate biomedically relevant generalizations, such as a model's ability to generalize to entirely unseen compounds, or to granularly resolve patient response to a polytherapy.

## TDC Data Loaders

Each machine learning problem comprises of multiple learning tasks. TDC provides a data loader class for each *task* inheriting from the base data loader. 

Each learning task is instantiated through many datasets. Most datasets for a given task have the same structure. To get a dataset, use the `dataset_name` as a function input to the task data loader.

Suppose you want to retrieve *dataset* `X` to study therapeutics *task* `Y` which falls under the *problem* `Z`. To obtain the dataset and its associated data split, use the following:

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

See all therapeutic tasks and datasets on the [TDC website](https://zitniklab.hms.harvard.edu/TDC/overview/)!

## TDC Data Functions

#### Data Split

To retrieve the training/validation/test dataset split, you could simply type
```python 
data = X(name = Y)
data.get_split(seed = 42)
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


## TDC Leaderboards

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

TDC is designed to be a community-driven effort. Checkout the [contribution guide](CONTRIBUTE.md) to contribute!

## Contact

Send emails to [us](mailto:kexinhuang@hsph.harvard.edu) or open an issue.

## Data Server Maintenance Issues

TDC is hosted in [Harvard Dataverse](https://dataverse.harvard.edu/). When dataverse is under maintenance, TDC will not able to retrieve datasets. Although rare, when it happens, please come back in couple of hours or check the status by visiting the [dataverse website](https://dataverse.harvard.edu/).

## License
TDC codebase is under MIT license. For individual dataset usage, please refer to the dataset license in the website.
