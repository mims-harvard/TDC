import sys

sys.path.append("..")
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import functools
import matplotlib.pyplot as plt
import tensorflow as tf

from dqn import molecules
from dqn import deep_q_networks
from dqn.py.SA_Score import sascorer
from chemutil import similarity

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, QED

from tdc import Oracle

qed_oracle = Oracle(name="qed")
from tdc import Evaluator

diversity = Evaluator(name="Diversity")


import pyscreener
from tdc import Oracle

oracle2 = Oracle(
    name="Docking_Score",
    software="vina",
    pyscreener_path="./",
    receptors=["/project/molecular_data/graphnn/pyscreener/testing_inputs/DRD3.pdb"],
    center=(9, 22.5, 26),
    size=(15, 15, 15),
    buffer=10,
    path="./",
    num_worker=3,
    ncpu=8,
)


def get_file_step_number(filename):
    return int(filename.split("-")[1].split(".")[0])


ckpt_folder = "docking1"
input_smiles = "CO"
result_folder = "result"
step_list = [
    int(file.split(".")[0].split("-")[1])
    for file in os.listdir(ckpt_folder)
    if file[:4] == "ckpt" and file[-4:] == "meta"
]
file_list = [
    os.path.join(ckpt_folder, file[:-5])
    for file in os.listdir(ckpt_folder)
    if file[:4] == "ckpt" and file[-4:] == "meta"
]


def eval(model_file, input_smiles, epsilon=0.1):
    # hparams_file = os.path.join(model_dir, 'config.json')
    hparams_file = "./configs/naive_dqn.json"
    fh = open(hparams_file, "r")
    hp_dict = json.load(fh)
    hparams = deep_q_networks.get_hparams(**hp_dict)
    fh.close()

    environment = molecules.Molecule(
        atom_types=set(hparams.atom_types),
        init_mol=input_smiles,
        allow_removal=hparams.allow_removal,
        allow_no_modification=hparams.allow_no_modification,
        allowed_ring_sizes=set(hparams.allowed_ring_sizes),
        allow_bonds_between_rings=hparams.allow_bonds_between_rings,
        max_steps=hparams.max_steps_per_episode,
    )

    dqn = deep_q_networks.DeepQNetwork(
        input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
        q_fn=functools.partial(deep_q_networks.multi_layer_model, hparams=hparams),
        optimizer=hparams.optimizer,
        grad_clipping=hparams.grad_clipping,
        num_bootstrap_heads=hparams.num_bootstrap_heads,
        gamma=hparams.gamma,
        epsilon=0.0,
    )

    tf.reset_default_graph()
    with tf.Session() as sess:
        dqn.build()

        model_saver = tf.train.Saver(max_to_keep=hparams.max_num_checkpoints)
        model_saver.restore(sess, model_file)
        smiles_lst = []

        environment.initialize()
        for step in range(hparams.max_steps_per_episode):
            steps_left = hparams.max_steps_per_episode - environment.num_steps_taken

            if hparams.num_bootstrap_heads:
                head = np.random.randint(hparams.num_bootstrap_heads)
            else:
                head = 0
            valid_actions = list(environment.get_valid_actions())
            observations = np.vstack(
                [
                    np.append(deep_q_networks.get_fingerprint(act, hparams), steps_left)
                    for act in valid_actions
                ]
            )

            action = valid_actions[
                dqn.get_action(observations, head=head, update_epsilon=epsilon)
            ]
            result = environment.step(action)
    return result.state


for file in tqdm(file_list):
    smiles_lst = []
    for i in range(100):
        smiles = eval(file, input_smiles)
        score = oracle2(smiles)
        smiles_lst.append((smiles, score))
    result_file = os.path.join(result_folder, file.split("/")[-1])
    with open(result_file, "w") as fout:
        for smiles, score in smiles_lst:
            fout.write(smiles + "\t" + str(score) + "\n")
