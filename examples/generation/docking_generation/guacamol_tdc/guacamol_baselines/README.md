# GuacaMol Baselines

A series of baseline model implementations for the [`guacamol`](https://github.com/BenevolentAI/guacamol) benchmark 
for generative chemistry.  
A more in depth explanation of the benchmarks and scores for these baselines is 
can be found in our [paper](https://arxiv.org/abs/1811.09621).

## Dependencies
To install all dependencies:
```bash
conda install rdkit -c rdkit
pip install -r requirements.txt
```


## Dataset
Some baselines require the `guacamol` dataset to run, to get it run:
```bash
bash fetch_guacamol_dataset.sh
```


## Random Sampler
Dummy baseline, always returning random molecules form the `guacamol` training set.

To execute the goal-directed generation benchmarks:
```bash
python -m random_smiles_sampler.goal_directed_generation
```

To execute the distribution learning benchmarks:
```bash
python -m random_smiles_sampler.distribution_learning
```


## Best from ChEMBL
Dummy baseline that simply returns the molecules from the `guacamol` 
training set that best satisfy the score of a goal-directed benchmark.  
There is no model nor training, its only purpose is to establish a lower bound
on the benchmark scores.

To execute the goal-directed generation benchmarks:
```bash
python -m best_from_chembl.goal_directed_generation
```

No distribution learning benchmark available.


## SMILES GA
Genetic algorithm on SMILES as described in: https://www.journal.csj.jp/doi/10.1246/cl.180665  

Implementation adapted from: https://github.com/tsudalab/ChemGE

To execute the goal-directed generation benchmarks:
```bash
python -m smiles_ga.goal_directed_generation
```

No distribution learning benchmark available.


## Graph GA
Genetic algoritm on molecule graphs as described in: https://doi.org/10.26434/chemrxiv.7240751  

Implementation adapted from: https://github.com/jensengroup/GB-GA  

To execute the goal-directed generation benchmarks:
```bash
python -m graph_ga.goal_directed_generation
```

No distribution learning benchmark available.


## Graph MCTS
Monte Carlo Tree Search on molecule graphs as described in: https://doi.org/10.26434/chemrxiv.7240751  

Implementation adapted from: https://github.com/jensengroup/GB-GB  

To execute the goal-directed generation benchmarks:
```bash
python -m graph_mcts.goal_directed_generation
```

To execute the distribution learning benchmarks:
```bash
python -m graph_mcts.distribution_learning
```

To re-generate the distribution statistics as pickle files:
```bash
python -m graph_mcts.analyze_dataset
```


## SMILES LSTM Hill Climbing
Long-short term memory on SMILES as described in: https://arxiv.org/abs/1701.01329  

This implementation optimizes using *hill climbing* algorithm.  

Implementation by [BenevolentAI](https://benevolent.ai/)

A pre-trained model is provided in: [smiles_lstm/pretrained_model](https://github.com/BenevolentAI/guacamol_baselines/tree/master/smiles_lstm_hc/pretrained_model)  

To execute the goal-directed generation benchmarks: 
```bash
python -m smiles_lstm_hc.goal_directed_generation
```

To execute the distribution learning benchmark:
```bash
python -m smiles_lstm_hc.distribution_learning
```

To train a model from scratch:
```bash
python -m smiles_lstm_hc.train_smiles_lstm_model
```

## SMILES LSTM PPO
Long-short term memory on SMILES as described in: https://arxiv.org/abs/1701.01329  

This implementation optimizes using [*proximal policy optimization*](https://arxiv.org/pdf/1707.06347.pdf) algorithm.  

Implementation by [BenevolentAI](https://benevolent.ai/)

A pre-trained model is provided in: [smiles_lstm/pretrained_model](https://github.com/BenevolentAI/guacamol_baselines/tree/master/smiles_lstm_ppo/pretrained_model)  

To execute the goal-directed generation benchmarks: 
```bash
python -m smiles_lstm_ppo.goal_directed_generation
```

