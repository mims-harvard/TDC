## Drug Combination Benchmark Group MLP Baselines

This directory contains the code used to train a simple baseline MLP regression model for the DrugCombo benchmark. 

To run the model, simply:

```python
python train_MLP.py --epochs 10 --batch_size 128 --cuda True

'''
{'drugcomb_css': [16.858, 0.005], 'drugcomb_css_kidney': [14.57, 0.003], 'drugcomb_css_lung': [15.653, 0.017], 'drugcomb_css_breast': [13.432, 0.049], 'drugcomb_css_hematopoietic_lymphoid': [28.764, 0.201], 'drugcomb_css_colon': [17.729, 0.042], 'drugcomb_css_prostate': [15.692, 0.005], 'drugcomb_css_ovary': [15.263, 0.041], 'drugcomb_css_skin': [15.663, 0.065], 'drugcomb_css_brain': [15.694, 0.006], 'drugcomb_hsa': [4.453, 0.002], 'drugcomb_loewe': [9.184, 0.001], 'drugcomb_bliss': [4.56, 0.0], 'drugcomb_zip': [4.027, 0.003]}
'''

```

