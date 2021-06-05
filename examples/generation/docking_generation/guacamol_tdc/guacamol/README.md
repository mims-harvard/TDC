# GuacaMol

[![Build Status](https://travis-ci.com/BenevolentAI/guacamol.svg?branch=master)](https://travis-ci.com/BenevolentAI/guacamol)

**GuacaMol** is an open source Python package for benchmarking of models for 
*de novo* molecular design.

For an in-depth explanation of the types of benchmarks and baseline scores,
please consult our paper 
[Benchmarking Models for De Novo Molecular Design](https://arxiv.org/abs/1811.09621)

## Installation

The easiest way to install `guacamol` is with `pip`:
```bash
pip install guacamol
```

`guacamol` requires the [RDKit library](http://rdkit.org/) (version `2018.09.1.0` or newer).


## Benchmarking models

For the distribution-learning benchmarks, specialize `DistributionMatchingGenerator` 
(from `guacamol.distribution_matching_generator`) for your model. 
Instances of this class must be able to generate molecules similar to the training set.  
For the actual benchmarks, call `assess_distribution_learning` 
(from `guacamol.assess_distribution_learning`) with an instance of your class. 
You must also provide the location of the training set file (See section "Data" below).

For the goal-directed benchmarks, specialize `GoalDirectedGenerator` 
(from `guacamol.goal_directed_generator`) for your model. 
Instances of this class must be able to generate a specified number of molecules 
that achieve high scores for a given scoring function.  
For the actual benchmarks, call `assess_goal_directed_generation` 
(from `guacamol.assess_goal_directed_generation`) with an instance of your class. 

Example implementations for baseline methods are available from https://github.com/BenevolentAI/guacamol_baselines.


## Data

For fairness in the evaluation of the benchmarks and comparability of the results, 
you should use a training set containing molecules from the ChEMBL dataset.
Follow the procedure described below to get standardized datasets.


### Download

You can download pre-built datasets [here](https://figshare.com/projects/GuacaMol/56639):

md5 `05ad85d871958a05c02ab51a4fde8530` [training](https://ndownloader.figshare.com/files/13612760 )  
md5 `e53db4bff7dc4784123ae6df72e3b1f0` [validation](https://ndownloader.figshare.com/files/13612766)  
md5 `677b757ccec4809febd83850b43e1616` [test](https://ndownloader.figshare.com/files/13612757)  
md5 `7d45bc95c33c10cb96ef5e78c38ac0b6` [all](https://ndownloader.figshare.com/files/13612745)  


### Generation

To generate the training data yourself, run 
```
python -m guacamol.data.get_data -o [output_directory]
```
which will download and process ChEMBL for you in your current folder.

This script will use the molecules from 
[`holdout_set_gcm_v1.smiles`](https://github.com/BenevolentAI/guacamol/blob/master/guacamol/data/holdout_set_gcm_v1.smiles)
as a holdout set, and will exclude molecules very similar to these.

Different versions of your Python packages may lead to differences in the generated dataset, which will cause the script to fail.
See the section below ("Docker") to reproducibly generate the standardized dataset with the hashes given above.

### Docker

To be sure that you have the right dependencies you can build a Docker image, run from the top-level directory:
```
docker build -t guacamol-deps -f dockers/Dockerfile .
```
Then you can run:
```
docker run --rm -it  -v `pwd`:/guacamol -w /guacamol guacamol-deps python -m guacamol.data.get_data -o guacamol/data
```


## Leaderboard

See [https://benevolent.ai/guacamol](https://benevolent.ai/guacamol).
