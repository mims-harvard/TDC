<p align="center"><img src="https://raw.githubusercontent.com/mims-harvard/TDC/master/fig/logo.png" alt="logo" width="600px" /></p>

----

[![website](https://img.shields.io/badge/website-live-brightgreen)](https://tdcommons.ai)
[![PyPI version](https://badge.fury.io/py/PyTDC.svg)](https://badge.fury.io/py/PyTDC)
[![Downloads](https://pepy.tech/badge/pytdc/month)](https://pepy.tech/project/pytdc)
[![Downloads](https://pepy.tech/badge/pytdc)](https://pepy.tech/project/pytdc)
[![GitHub Repo stars](https://img.shields.io/github/stars/mims-harvard/TDC)](https://github.com/mims-harvard/TDC/stargazers)
[![GitHub Repo stars](https://img.shields.io/github/forks/mims-harvard/TDC)](https://github.com/mims-harvard/TDC/network/members)
[![Build Status](https://travis-ci.org/mims-harvard/TDC.svg?branch=master)](https://travis-ci.org/mims-harvard/TDC)
[![TDC CircleCI](https://circleci.com/gh/mims-harvard/TDC.svg?style=svg)](https://app.circleci.com/pipelines/github/mims-harvard/TDC)

[**Project Website**](https://tdcommons.ai) | [**Paper**](https://arxiv.org/abs/2102.09548) | [**Slack**](https://join.slack.com/t/pytdc/shared_invite/zt-qjjtloo4-naM1gsab8Ikpb03qXI9xzQ) | [**TDC Mailing List**](https://groups.io/g/tdc) | [![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40ProjectTDC)](https://twitter.com/ProjectTDC)

**Therapeutics Data Commons (TDC)** is the first unifying framework to systematically access and evaluate machine learning across the entire range of therapeutics.

The collection of curated datasets, learning tasks, and benchmarks in TDC serves as a meeting point for domain and machine learning scientists. We envision that TDC can considerably accelerate machine-learning model development, validation and transition into biomedical and clinical implementation.

TDC is an open-source initiative. To get involved, check out the [Contribution Guide](CONTRIBUTE.md)!

**Invited talk at the [Harvard Symposium on Drugs for Future Pandemics (#futuretx20)](https://www.drugsymposium.org/)** [**\[Slides\]**](https://drive.google.com/file/d/11eTrh_lsqPcwu3RZRYjJGNpJ3s18YlBS/view) [**\[Video\]**](https://youtu.be/ZuCOhEZtaOw)


## Updates
- `0.2.0`: Release docking molecule generation benchmark! Checkout [here](https://tdcommons.ai/benchmark/docking_group/overview/)!
- `0.1.9`: Support molecule filters! Checkout [here](https://tdcommons.ai//functions/data_process/#molecule-filters)!
- `0.1.8`: Streamlined and simplified the leaderboard programming frameworks! Now, you can submit a result for a single dataset! Checkout [here](https://tdcommons.ai/benchmark/overview/)!
- TDC white paper is alive on [arXiv](https://arxiv.org/abs/2102.09548)!

<details>
  <summary>Click here for older updates!</summary>

- `0.1.6`: Released the second leaderboard on drug combination screening prediction! Checkout [here](https://tdcommons.ai/benchmark/drugcombo_group/)!
- `0.1.5`: Added four realistic oracles from docking scores and synthetic accessibility! Checkout [here](https://tdcommons.ai/functions/oracles/)!
- `0.1.4`: Added the 1st version of [`MolConvert`](https://tdcommons.ai/functions/data_process/#molecule-conversion) class that can map among ~15 molecular formats in 2 lines of code (For 2D: from SMILES/SEFLIES and convert to SELFIES/SMILES, Graph2D, PyG, DGL, ECFP2-6, MACCS, Daylight, RDKit2D, Morgan, PubChem; For 3D: from XYZ, SDF files to Graph3D, Columb Matrix); Also a quality check on DTI datasets with IDs added.
- Checkout **[Contribution Guide](CONTRIBUTE.md)** to add new dataset, task, function!
- `0.1.3`: Added new therapeutics task on CRISPR Repair Outcome Prediction! Added a data function to map molecule to popular cheminformatics fingerprint.
- `0.1.2`: The first TDC Leaderboard is released! Checkout the leaderboard guide [here](https://tdcommons.ai/benchmark/overview/) and the ADMET Leaderboard [here](https://tdcommons.ai/benchmark/admet_group/).
- `0.1.1`: Replaced VD, Half Life and Clearance datasets from new sources that have higher qualities. Added LD50 to Tox.
- `0.1.0`: Molecule quality check for ADME, Toxicity and HTS (canonicalized, and remove error mols).
- `0.0.9`: Added DrugComb NCI-60, CYP2C9/2D6/3A4 substrates, Carcinogens toxicity! 
- `0.0.8`: Added hREG, DILI, Skin Reaction, Ames Mutagenicity, PPBR from AstraZeneca; added meta oracles!

</details>


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
numpy, pandas, tqdm, scikit-learn, fuzzywuzzy, seaborn
```

For utilities requiring extra dependencies, TDC prints installation instructions. To install full dependencies, please use the following `conda-forge` solution. 

### Using `conda`

Data functions for molecule oracles, scaffold split, etc., require certain packages like RDKit. To install those packages, use the following `conda` installation: 

```bash
conda install -c conda-forge pytdc
```


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
| [106](tutorials/TDC_106_BenchmarkGroup_Submission_Demo.ipynb)   | Benchmark submission                             |
| [DGL](tutorials/DGL_User_Group_Demo.ipynb)   | Demo for DGL GNN User Group Meeting                             |


## Design of TDC

TDC has a unique three-tiered hierarchical structure, which to our knowledge, is the first attempt at systematically organizing machine learning for therapeutics. We organize TDC into three distinct *problems*. For each problem, we give a collection *learning tasks*. Finally, for each task, we provide a series of *datasets*.

In the first tier, after observing a large set of therapeutics tasks, we categorize and abstract out three major areas (i.e., problems) where machine learning can facilitate scientific advances, namely, single-instance prediction, multi-instance prediction, and generation:

* Single-instance prediction `single_pred`: Prediction of property given individual biomedical entity.
* Multi-instance prediction `multi_pred`: Prediction of property given multiple biomedical entities. 
* Generation `generation`: Generation of new desirable biomedical entities.

<p align="center"><img src="https://raw.githubusercontent.com/mims-harvard/TDC/master/fig/tdc_problems.png" alt="problems" width="500px" /></p>

The second tier in the TDC structure is organized into learning tasks. Improvement on these tasks can result in numerous applications, including identifying personalized combinatorial therapies, designing novel class of antibodies, improving disease diagnosis, and finding new cures for emerging diseases.

Finally, in the third tier of TDC, each task is instantiated via multiple datasets. For each dataset, we provide several splits of the dataset into training, validation, and test sets to simulate the type of understanding and generalization (e.g., the model's ability to generalize to entirely unseen compounds or to granularly resolve patient response to a polytherapy) needed for transition into production and clinical implementation.


## TDC Data Loaders

TDC provides a collection of workflows with intuitive, high-level APIs for both beginners and experts to create machine learning models in Python. Building off the modularized "Problem--Learning Task--Data Set" structure (see above) in TDC, we provide a three-layer API to access any learning task and dataset. This hierarchical API design allows us to easily incorporate new tasks and datasets.

For a concrete example, to obtain the HIA dataset from ADME therapeutic learning task in the single-instance prediction problem:

```python
from tdc.single_pred import ADME
data = ADME(name = 'HIA_Hou')
# split into train/val/test with scaffold split methods
split = data.get_split(method = 'scaffold')
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
# {'train': df_train, 'val': df_val, 'test': df_test}
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

## TDC Leaderboards

Each dataset in TDC is a benchmark, and we provide training/validation and test sets for it, together with data splits and performance evaluation metrics. To participate in the leaderboard for a specific benchmark, follow these steps:

* Use the TDC benchmark data loader to retrieve the benchmark.

* Use training and/or validation set to train your model.

* Use the TDC model evaluator to calculate the performance of your model on the test set.

* Submit the test set performance to a TDC leaderboard.

As many datasets share a therapeutics theme, we organize specific benchmarks into meaningfully defined groups, referred to as benchmark groups. Datasets and tasks within a benchmark group are carefully curated and centered around a theme (for example, prediction of ADMET properties). While each benchmark group consists of multiple benchmarks, you can submit each dataset/benchmark result separately. Here is the code framework to access the benchmarks:

```python
from tdc import BenchmarkGroup
group = BenchmarkGroup(name = 'ADMET_Group', path = 'data/')
predictions_list = []

for seed in [1, 2, 3, 4, 5]:
    benchmark = group.get('Caco2_Wang') 
    # all benchmark names in a benchmark group are stored in group.dataset_names
    predictions = {}
    name = benchmark['name']
    train_val, test = benchmark['train_val'], benchmark['test']
    train, valid = group.get_train_valid_split(benchmark = name, split_type = 'default', seed = seed)
    
        # --------------------------------------------- # 
        #  Train your model using train, valid, test    #
        #  Save test prediction in y_pred_test variable #
        # --------------------------------------------- #
        
    predictions[name] = y_pred_test
    predictions_list.append(predictions)

results = group.evaluate_many(predictions_list)
# {'caco2_wang': [6.328, 0.101]}
```

For more information, please visit [here](https://tdcommons.ai/benchmark/overview/).


## Cite Us

If you found our work useful, please cite us:

```
@article{tdc,
  title={Therapeutics Data Commons: Machine Learning Datasets and Tasks for Therapeutics},
  author={Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Roohani, Yusuf and Leskovec, Jure and Coley, Connor W and Xiao, Cao and Sun, Jimeng and Zitnik, Marinka},
  journal={arXiv preprint arXiv:2102.09548},
  year={2021}
}
```

## Contribute

TDC is an open-source community-driven effort. If you want to get involved, join the [Slack Workspace](https://join.slack.com/t/pytdc/shared_invite/zt-qjjtloo4-naM1gsab8Ikpb03qXI9xzQ) and checkout the [contribution guide](CONTRIBUTE.md)!

## Contact

Send emails to [us](mailto:contact@tdcommons.ai) or open an issue.

## Data Server Maintenance Issues

TDC is hosted in [Harvard Dataverse](https://dataverse.harvard.edu/). When dataverse is under maintenance, TDC will not able to retrieve datasets. Although rare, when it happens, please come back in couple of hours or check the status by visiting the [dataverse website](https://dataverse.harvard.edu/).

## License
TDC codebase is under MIT license. For individual dataset usage, please refer to the dataset license found in the website.
