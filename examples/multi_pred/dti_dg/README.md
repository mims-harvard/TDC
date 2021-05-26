# TDC DTI Domain Generalization Leaderboard

We adapt code from [domainbed](https://arxiv.org/abs/2007.01434) to add 7 baselines for this leaderboard. For the backbone model, we use [DeepDTA](https://academic.oup.com/bioinformatics/article/34/17/i821/5093245), one of the SOTA baselines for DTI affinity prediction. 

### Environment

`torch, numpy, pandas, tqdm, scikit-learn`

### Run

```python
cd domainbed/
python train.py --algorithm GroupDRO --seed 0

# supported model: ERM/IRM/GroupDRO/MMD/CORAL/AndMask/MTL
```


### Add your own domain generalization algorithm

Go to `domainbed/algorithm.py` script to add your algorithm. 