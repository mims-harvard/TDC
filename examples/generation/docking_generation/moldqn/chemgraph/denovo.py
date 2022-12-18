import sys

sys.path.append("..")
import os
import json
import numpy as np
import pandas as pd
import functools
from dqn import molecules
from dqn import deep_q_networks
from dqn.py.SA_Score import sascorer
from chemutil import similarity

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, QED

from tdc import Oracle

qed_oracle = Oracle(name="qed")

# import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path


def latest_ckpt(path):
    return max(
        [int(p.stem.split("-")[1]) for p in path.iterdir() if p.stem[:4] == "ckpt"]
    )


# basepath = '/Users/odin/sherlock_scratch/moldqn2/target_sas/mol%i_target_%.1f'
path = Path("save_qed")
# path = Path(basepath %(1, 4.8))
latest_ckpt(path)

all_molecules = ["CO", "O", "C", "N"]


def eval(model_dir, idx):
    ckpt = latest_ckpt(Path(model_dir))
    hparams_file = "./configs/naive_dqn.json"
    fh = open(hparams_file, "r")
    hp_dict = json.load(fh)
    hparams = deep_q_networks.get_hparams(**hp_dict)
    fh.close()

    environment = molecules.Molecule(
        atom_types=set(hparams.atom_types),
        init_mol=all_molecules[idx],
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
        model_saver.restore(sess, os.path.join(model_dir, "ckpt-%i" % ckpt))

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

            for epsilon in range(100):
                epsilon = epsilon / 10000
                action = valid_actions[
                    dqn.get_action(observations, head=head, update_epsilon=epsilon)
                ]
                result = environment.step(action)
                print("epsilon", epsilon, result.state)

            # action = valid_actions[dqn.get_action(
            #     observations, head=head, update_epsilon=0.0)]
            # result = environment.step(action)
    return ckpt, result


all_results = []


for i in range(len(all_molecules)):
    ckpt, result = eval(path, i)
    ori_qed = qed_oracle(all_molecules[i])
    qed = qed_oracle(result.state)
    sim = similarity(all_molecules[i], result.state)
    all_results.append((i, ckpt, all_molecules[i], result.state, ori_qed, qed))
    # print(all_molecules[i], result.state, ori_qed, qed)
