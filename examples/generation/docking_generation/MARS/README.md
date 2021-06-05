# MARS: Markov Molecular Sampling for Multi-objective Drug Discovery

Thanks for your interest! This is the code repository for our ICLR 2021 paper [MARS: Markov Molecular Sampling for Multi-objective Drug Discovery](https://openreview.net/pdf?id=kHSu4ebxFXY). 

## Dependencies

The `conda` environment is exported as `environment.yml`. You can also manually install these packages:

```bash
conda install -c conda-forge rdkit
conda install tqdm tensorboard scikit-learn
conda install pytorch cudatoolkit=11.1 -c pytorch -c conda-forge
conda install -c dglteam dgl-cuda11.1

# for cpu only
conda install pytorch cpuonly -c pytorch
conda install -c dglteam dgl
```

## Run

> Note: Run the commands **outside** the `MARS` directory.

To extract molecular fragments from a database:


```bash 
python preprocess_zinc.py
```
output is `data/zinc.txt`

```bash
python -m MARS.datasets.prepro_vocab
```



To sample molecules:

```bash
cd /project/molecular_data/graphnn/MARS

source activate MARS 

rm -rf MARS/runs/try;

python -m MARS.main --run_dir runs/try
```

## docking 

```bash
cd /project/molecular_data/graphnn/MARS

source activate MARS 

export PATH=$PATH:/project/molecular_data/graphnn/docking/ADFRsuite_installed_directory/bin

export PATH=$PATH:/project/molecular_data/graphnn/docking/autodock_vina_1_1_2_linux_x86/bin


./upload.sh main.py 
./upload.sh sampler.py 
./upload.sh estimator 
```

```
rm -rf MARS/runs/try4;  python -m MARS.main --run_dir runs/try4
```



```python
import pyscreener
from tdc import Oracle 
oracle2 = Oracle(name = 'Docking_Score', software='vina', pyscreener_path = './', pdbids=['5WIU'], center=(-18.2, 14.4, -16.1), size=(15.4, 13.9, 14.5), buffer=10, path='./', num_worker=1, ncpu=4)
```