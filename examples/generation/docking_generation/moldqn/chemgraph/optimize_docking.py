# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Optimizes QED of a molecule with DQN.

This experiment tries to find the molecule with the highest QED
starting from a given molecule.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os


from absl import app
from absl import flags

from rdkit import Chem

from rdkit.Chem import QED
from tensorflow.compat.v1 import gfile

from dqn import deep_q_networks
from dqn import molecules as molecules_mdp
from dqn import run_dqn
from dqn.tensorflow_core import core


FLAGS = flags.FLAGS

import pickle
import collections
import pyscreener
from tdc import Oracle

# oracle2 = Oracle(name = 'Docking_Score', software='vina', pyscreener_path = './', pdbids=['5WIU'], center=(-18.2, 14.4, -16.1), size=(15.4, 13.9, 14.5), buffer=10, path='./', num_worker=1, ncpu=4)
# oracle2 = Oracle(name = 'Docking_Score', software='vina', pyscreener_path = './', pdbids=['DRD3'], center=(-18.2, 14.4, -16.1), size=(15.4, 13.9, 14.5), buffer=10, path='./', num_worker=1, ncpu=4)

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
# exit()

run = 9


class Result(collections.namedtuple("Result", ["state", "reward", "terminated"])):
    """A namedtuple defines the result of a step for the molecule class.

    The namedtuple contains the following fields:
      state: Chem.RWMol. The molecule reached after taking the action.
      reward: Float. The reward get after taking the action.
      terminated: Boolean. Whether this episode is terminated.
    """


class DockingRewardMolecule(molecules_mdp.Molecule):
    """The molecule whose reward is the QED."""

    def __init__(self, discount_factor, **kwargs):
        """Initializes the class.

        Args:
          discount_factor: Float. The discount factor. We only
            care about the molecule at the end of modification.
            In order to prevent a myopic decision, we discount
            the reward at each step by a factor of
            discount_factor ** num_steps_left,
            this encourages exploration with emphasis on long term rewards.
          **kwargs: The keyword arguments passed to the base class.
        """
        super(DockingRewardMolecule, self).__init__(**kwargs)
        self.discount_factor = discount_factor

    def _reward(self):
        """Reward of a state.

        Returns:
          Float. QED of the current state.
        """
        # molecule = Chem.MolFromSmiles(self._state)
        # if molecule is None:
        #   return 0.0
        # qed = QED.qed(molecule)
        try:
            score = -oracle2(self._state)
        except:
            score = 0.0
        return score * self.discount_factor ** (self.max_steps - self.num_steps_taken)

    def step(self, action):
        """Takes a step forward according to the action.

        Args:
          action: Chem.RWMol. The action is actually the target of the modification.

        Returns:
          results: Namedtuple containing the following fields:
            * state: The molecule reached after taking the action.
            * reward: The reward get after taking the action.
            * terminated: Whether this episode is terminated.

        Raises:
          ValueError: If the number of steps taken exceeds the preset max_steps, or
            the action is not in the set of valid_actions.

        """
        if self._counter >= self.max_steps or self._goal_reached():
            raise ValueError("This episode is terminated.")
        if action not in self._valid_actions:
            raise ValueError("Invalid action.")
        self._state = action
        if self.record_path:
            self._path.append(self._state)
        self._valid_actions = self.get_valid_actions(force_rebuild=True)
        self._counter += 1

        reward = self._reward() if self._counter == self.max_steps - 1 else 0.0

        ### save
        if self._counter == self.max_steps - 1:
            try:
                self.smiles_value_lst.append((self._state, reward))
            except:
                self.smiles_value_lst = [(self._state, reward)]
            if len(self.smiles_value_lst) % 10 == 0:
                pickle.dump(
                    self.smiles_value_lst,
                    open("docking_smilesvaluelst_" + str(run) + ".pkl", "wb"),
                )
        ### save

        result = Result(
            state=self._state,
            reward=reward,
            terminated=(self._counter >= self.max_steps) or self._goal_reached(),
        )

        return result


def main(argv):
    del argv  # unused.
    if FLAGS.hparams is not None:
        with gfile.Open(FLAGS.hparams, "r") as f:
            hparams = deep_q_networks.get_hparams(**json.load(f))
    else:
        hparams = deep_q_networks.get_hparams()

    environment = DockingRewardMolecule(
        discount_factor=hparams.discount_factor,
        atom_types=set(hparams.atom_types),
        init_mol=FLAGS.start_molecule,
        allow_removal=hparams.allow_removal,
        allow_no_modification=hparams.allow_no_modification,
        allow_bonds_between_rings=hparams.allow_bonds_between_rings,
        allowed_ring_sizes=set(hparams.allowed_ring_sizes),
        max_steps=hparams.max_steps_per_episode,
    )

    dqn = deep_q_networks.DeepQNetwork(
        input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
        q_fn=functools.partial(deep_q_networks.multi_layer_model, hparams=hparams),
        optimizer=hparams.optimizer,
        grad_clipping=hparams.grad_clipping,
        num_bootstrap_heads=hparams.num_bootstrap_heads,
        gamma=hparams.gamma,
        epsilon=1.0,
    )

    run_dqn.run_training(hparams=hparams, environment=environment, dqn=dqn)

    core.write_hparams(hparams, os.path.join(FLAGS.model_dir, "config.json"))


if __name__ == "__main__":
    app.run(main)
