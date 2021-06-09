## ADMET Benchmark Group DeepPurpose Baselines

Paper: [https://doi.org/10.1093/bioinformatics/btaa1005](https://doi.org/10.1093/bioinformatics/btaa1005)

GitHub: [https://github.com/kexinhuang12345/DeepPurpose](https://github.com/kexinhuang12345/DeepPurpose)

In this directory, we show how to use DeepPurpose to build three models for ADMET predictions. It is roughly around 50 lines of codes for the entire 22 benchmarks with 5 random seeds in ADMET benchmark group.


## Installation

```bash
conda create -n DeepPurpose python=3.6
conda activate DeepPurpose
conda install -c conda-forge rdkit
pip install git+https://github.com/bp-kelley/descriptastorus
pip install DeepPurpose
pip install PyTDC
```

For build from source installation, checkout the [installation page](https://github.com/kexinhuang12345/DeepPurpose#install--usage) of DeepPurpose.

## Reproduce Results

```python
python run.py --model Morgan
```

You can select the following model in the --model parameter: 'Morgan', 'RDKit2D', 'CNN', 'NeuralFP', 'MPNN', 'AttentiveFP', 'AttrMasking', 'ContextPred'

## Sample Output

```python
python run.py --model RDKit2D
'''
{'caco2_wang': [0.393, 0.024],
 'hia_hou': [0.972, 0.008],
 'pgp_broccatelli': [0.918, 0.007],
 'bioavailability_ma': [0.672, 0.021],
 'lipophilicity_astrazeneca': [0.574, 0.017],
 'solubility_aqsoldb': [0.827, 0.047],
 'bbb_martins': [0.889, 0.016],
 'ppbr_az': [9.994, 0.319],
 'vdss_lombardo': [0.561, 0.025],
 'cyp2d6_veith': [0.616, 0.007],
 'cyp3a4_veith': [0.829, 0.007],
 'cyp2c9_veith': [0.742, 0.006],
 'cyp2d6_substrate_carbonmangels': [0.677, 0.047],
 'cyp3a4_substrate_carbonmangels': [0.639, 0.012],
 'cyp2c9_substrate_carbonmangels': [0.36, 0.04],
 'half_life_obach': [0.184, 0.111],
 'clearance_microsome_az': [0.586, 0.014],
 'clearance_hepatocyte_az': [0.382, 0.007],
 'herg': [0.841, 0.02],
 'ames': [0.823, 0.011],
 'dili': [0.875, 0.019],
 'ld50_zhu': [0.678, 0.003]}
'''
```
## Contact

Please contact [Kexin](mailto:kexinhuang@hsph.harvard.edu) if you have any question!
