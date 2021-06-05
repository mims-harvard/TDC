# GCPN docking for TDC leaderboard 


## conda

```bash
source activate gcpn
```

## Installation
- Install rdkit, please refer to the offical website for further details, using anaconda is recommended:
```bash
conda create -c rdkit -n my-rdkit-env rdkit
```
- Install mpi4py, networkx:
```bash
conda install mpi4py
pip install networkx=1.11
```
- Install OpenAI baseline dependencies:
```bash
cd rl-baselines
pip install -e .
```
- Install customized molecule gym environment:
```bash
cd gym-molecule
pip install -e .
```


## Code description
There are 4 important files:
- `run_molecule.py` is the main code for running the program. You may tune all kinds of hyper-parameters there.
- The molecule environment code is in `gym-molecule/gym_molecule/envs/molecule.py`.
- RL related code is in `rl-baselines/baselines/ppo1` folder: `gcn_policy.py` is the GCN policy network; `pposgd_simple_gcn.py` is the PPO algorithm specifically tuned for GCN policy.

## Run docking 

```bash
source activate gcpn

cd /project/molecular_data/graphnn/GCPN

rm -rf ckpt result oracle_call_cnt

mkdir ckpt result

export PATH=$PATH:/project/molecular_data/graphnn/mol_dqn_docking/package_install/ADFRsuite_x86_64Linux_1.0/bin

export PATH=$PATH:/project/molecular_data/graphnn/mol_dqn_docking/package_install/autodock_vina_1_1_2_linux_x86/bin

python run_molecule.py 
```

## Run
- single process run
```bash
python run_molecule.py
```
- mutiple processes run
```bash
mpirun -np 8 python run_molecule.py 2>/dev/null
```
`2>/dev/null` will hide the warning info provided by rdkit package.

We highly recommend using tensorboard to monitor the training process. To do this, you may run
```bash
tensorboard --logdir runs
```

All the generated molecules along the training process are stored in the `molecule_gen` folder, each run configuration is stored in a different csv file. Molecules are stored using SMILES strings, along with the desired properties scores.






