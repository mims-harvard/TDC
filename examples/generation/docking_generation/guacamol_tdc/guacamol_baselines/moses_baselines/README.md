# Wrappers for baseline models from MOSES

MOSES (https://github.com/molecularsets/moses) provides baseline implementations of a series of generative models:
* VAE
* ORGAN
* AAE
* JT-VAE
* charRNN

Here, we wrap those models for applying the GuacaMol benchmarks.
Since MOSES does not consider goal-directed optimization, only distribution-learning benchmarks are done with these models.

JT-VAE is not included because training it with the ChEMBL dataset leads to errors.
Also, `charRNN` is not included because it is similar to the SMILES LSTM model.


## Execution

Execute the following commands from the root of the repository.
Replace `--device cpu` by `--device cuda` if your machine is GPU-enabled.

### AAE

Train:
```bash
python -m moses_baselines.aae_train --device cpu --train_load data/guacamol_v1_train.smiles
```

Benchmark:
```bash
python -m moses_baselines.aae_distribution_learning --device cpu --n_samples 0
```

### VAE

Train:
```bash
python -m moses_baselines.vae_train --device cpu --train_load data/guacamol_v1_train.smiles
```

Benchmark:
```bash
python -m moses_baselines.vae_distribution_learning --device cpu --n_samples 0
```

### ORGAN

Train:
```bash
python -m moses_baselines.organ_train --device cpu --train_load data/guacamol_v1_train.smiles
```

Benchmark:
```bash
python -m moses_baselines.organ_distribution_learning --device cpu --n_samples 0
```
