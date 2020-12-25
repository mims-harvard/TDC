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

### Morgan Fingerprint + MLP

```python
python run.py --model Morgan
'''
{'caco2_wang': [0.908, 0.06],
 'hia_hou': [0.807, 0.072],
 'pgp_broccatelli': [0.88, 0.006],
 'bioavailability_ma': [0.581, 0.086],
 'lipophilicity_astrazeneca': [0.701, 0.009],
 'solubility_aqsoldb': [1.203, 0.019],
 'bbb_martins': [0.823, 0.015],
 'ppbr_az': [12.848, 0.362],
 'vdss_lombardo': [0.493, 0.011],
 'cyp2d6_veith': [0.587, 0.011],
 'cyp3a4_veith': [0.827, 0.009],
 'cyp2c9_veith': [0.715, 0.004],
 'cyp2d6_substrate_carbonmangels': [0.671, 0.066],
 'cyp3a4_substrate_carbonmangels': [0.633, 0.013],
 'cyp2c9_substrate_carbonmangels': [0.38, 0.015],
 'half_life_obach': [0.329, 0.083],
 'clearance_microsome_az': [0.492, 0.02],
 'clearance_hepatocyte_az': [0.272, 0.068],
 'herg': [0.736, 0.023],
 'ames': [0.794, 0.008],
 'dili': [0.832, 0.021],
 'ld50_zhu': [0.649, 0.019]}
'''
```

### RDKit2D Fingerprint + MLP

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

### CNN 1D on SMILES

```python
python run.py --model CNN
'''
{'caco2_wang': [0.446, 0.036],
 'hia_hou': [0.869, 0.026],
 'pgp_broccatelli': [0.908, 0.012],
 'bioavailability_ma': [0.613, 0.013],
 'lipophilicity_astrazeneca': [0.743, 0.02],
 'solubility_aqsoldb': [1.023, 0.023],
 'bbb_martins': [0.781, 0.03],
 'ppbr_az': [11.106, 0.358],
 'vdss_lombardo': [0.226, 0.114],
 'cyp2d6_veith': [0.544, 0.053],
 'cyp3a4_veith': [0.821, 0.003],
 'cyp2c9_veith': [0.713, 0.006],
 'cyp2d6_substrate_carbonmangels': [0.485, 0.037],
 'cyp3a4_substrate_carbonmangels': [0.662, 0.031],
 'cyp2c9_substrate_carbonmangels': [0.367, 0.059],
 'half_life_obach': [0.038, 0.138],
 'clearance_microsome_az': [0.252, 0.116],
 'clearance_hepatocyte_az': [0.235, 0.021],
 'herg': [0.754, 0.037],
 'ames': [0.776, 0.015],
 'dili': [0.792, 0.016],
 'ld50_zhu': [0.675, 0.011]}
'''
```

## Contact

Please contact [Kexin](mailto:kexinhuang@hsph.harvard.edu) if you have any question!