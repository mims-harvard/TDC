# guacamol for tdc



## limited oracle evaluation 

modify the function generate_optimized_molecules(.., scoring_function, ..)

"max_oracle_num = 10000"  in scoring_function.py  


	best_from_chembl --- done 

	graph_ga ---- done 

	graph_mcts --- done, but confused about max_children 

	random_smiles_sampler --- done (no change)

	smiles_ga --- done 

## tested oracle 

1. default='v3'

2. goal_directed_suite_V in ./guacamol/ 


## docking oracle 


### count docking number 

`*.py`

```python

```

## run 

## guacamol 

oracle init and call: `guacamol/guacamol/common_scoring_functions.py`


## guacamol baseline 

(1) SMILES-LSTM `guacamol_baselines/smiles_lstm_hc/goal_directed_generation.py`
(2) Graph-GA `guacamol_baselines/graph_ga/goal_directed_generation.py`


## local 
```bash
cd /Users/futianfan/Downloads/spring2021/guacamol_tdc
./upload.sh 
```




## server 


### install guacamol 
```bash
cd /project/molecular_data/graphnn/guacamol_tdc/guacamol 
python setup.py install 
```


### graph ga
```bash
source activate guacamol_baselines
export PATH=$PATH:/project/molecular_data/graphnn/docking/ADFRsuite_installed_directory/bin
export PATH=$PATH:/project/molecular_data/graphnn/docking/autodock_vina_1_1_2_linux_x86/bin
cd /project/molecular_data/graphnn/pyscreener 

python graph_ga_run.py  

```


### smiles lstm 
```bash
cd /project/molecular_data/graphnn/guacamol_tdc/guacamol 
python setup.py install 


source activate guacamol_baselines
export PATH=$PATH:/project/molecular_data/graphnn/docking/ADFRsuite_installed_directory/bin
export PATH=$PATH:/project/molecular_data/graphnn/docking/autodock_vina_1_1_2_linux_x86/bin
cd /project/molecular_data/graphnn/pyscreener 

python run_smiles_lstm_hc.py  

```


### result analysis

```bash
cd /Users/futianfan/Downloads/spring2021/guacamol_tdc
./download.sh 
python results/analysis.py 
```

pls see `results/smiles_lstm_hc.png` for results. 














