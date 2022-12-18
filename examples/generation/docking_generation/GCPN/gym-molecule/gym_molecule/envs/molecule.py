import gym
import itertools
import numpy as np
from rdkit import Chem  # TODO(Bowen): remove and just use AllChem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog
import os

from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


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
    num_worker=1,
    ncpu=8,
)
smiles2docking = {}
# global oracle_call_cnt
# oracle_call_cnt = 0
oracle_call_file = "oracle_call_cnt"


def get_oracle_num():
    if not os.path.exists(oracle_call_file):
        return 0
    with open(oracle_call_file, "r") as fin:
        line = fin.readline().strip()
    return int(line)


def reset_oracle_num():
    with open(oracle_call_file, "w") as fout:
        fout.write("0")


reset_oracle_num()


def update_oracle_num():
    num = get_oracle_num()
    num += 1
    with open(oracle_call_file, "w") as fout:
        fout.write(str(num))


# import gym_molecule
import copy
import networkx as nx
from gym_molecule.envs.sascorer import calculateScore
from gym_molecule.dataset.dataset_utils import gdb_dataset, mol_to_nx, nx_to_mol
import random
import time
import matplotlib.pyplot as plt
import csv

from contextlib import contextmanager
import sys, os


# oracle_call = 0
global oracle_call

# block std out
@contextmanager
def nostdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# TODO(Bowen): check, esp if input is not radical
def convert_radical_electrons_to_hydrogens(mol):
    """
    Converts radical electrons in a molecule into bonds to hydrogens. Only
    use this if molecule is valid. Results a new mol object
    :param mol: rdkit mol object
    :return: rdkit mol object
    """
    m = copy.deepcopy(mol)
    if Chem.Descriptors.NumRadicalElectrons(m) == 0:  # not a radical
        return m
    else:  # a radical
        for a in m.GetAtoms():
            num_radical_e = a.GetNumRadicalElectrons()
            if num_radical_e > 0:
                a.SetNumRadicalElectrons(0)
                a.SetNumExplicitHs(num_radical_e)
    return m


def load_scaffold():
    cwd = os.path.dirname(__file__)
    path = os.path.join(os.path.dirname(cwd), "dataset", "vocab.txt")  # gdb 13
    with open(path, "r") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        data = [Chem.MolFromSmiles(row[0]) for row in reader]
        data = [
            mol
            for mol in data
            if mol.GetRingInfo().NumRings() == 1
            and (
                mol.GetRingInfo().IsAtomInRingOfSize(0, 5)
                or mol.GetRingInfo().IsAtomInRingOfSize(0, 6)
            )
        ]
        for mol in data:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        print("num of scaffolds:", len(data))
        return data


def load_conditional(type="low"):
    if type == "low":
        cwd = os.path.dirname(__file__)
        path = os.path.join(os.path.dirname(cwd), "dataset", "opt.test.logP-SA")
        import csv

        with open(path, "r") as fp:
            reader = csv.reader(fp, delimiter=" ", quotechar='"')
            data = [row + [id] for id, row in enumerate(reader)]
        # print(len(data))
        # print(data[799])
    elif type == "high":
        cwd = os.path.dirname(__file__)
        path = os.path.join(os.path.dirname(cwd), "dataset", "zinc_plogp_sorted.csv")
        import csv

        with open(path, "r") as fp:
            reader = csv.reader(fp, delimiter=",", quotechar='"')
            data = [[row[1], row[0], id] for id, row in enumerate(reader)]
            # data = [row for id, row in enumerate(reader)]
            data = data[0:800]
    return data


# data = load_conditional('low')
# data = load_conditional('high')
# print(data[799])


class MoleculeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        pass

    def init(
        self,
        data_type="zinc",
        logp_ratio=1,
        qed_ratio=1,
        sa_ratio=1,
        reward_step_total=1,
        is_normalize=0,
        reward_type="gan",
        reward_target=0.5,
        has_scaffold=False,
        has_feature=False,
        is_conditional=False,
        conditional="low",
        max_action=128,
        min_action=20,
        force_final=False,
    ):
        """
        own init function, since gym does not support passing argument
        """
        self.is_normalize = bool(is_normalize)
        self.is_conditional = is_conditional
        self.has_feature = has_feature
        self.reward_type = reward_type
        self.reward_target = reward_target
        self.force_final = force_final

        self.conditional_list = load_conditional(conditional)
        if self.is_conditional:
            self.conditional = random.sample(self.conditional_list, 1)[0]
            self.mol = Chem.RWMol(Chem.MolFromSmiles(self.conditional[0]))
            Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        else:
            self.mol = Chem.RWMol()
        self.smile_list = []
        if data_type == "gdb":
            possible_atoms = ["C", "N", "O", "S", "Cl"]  # gdb 13
        elif data_type == "zinc":
            possible_atoms = ["C", "N", "O", "S", "P", "F", "I", "Cl", "Br"]  # ZINC
        if self.has_feature:
            self.possible_formal_charge = np.array([-1, 0, 1])
            self.possible_implicit_valence = np.array([-1, 0, 1, 2, 3, 4])
            self.possible_ring_atom = np.array([True, False])
            self.possible_degree = np.array([0, 1, 2, 3, 4, 5, 6, 7])
            self.possible_hybridization = np.array(
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                ],
                dtype=object,
            )
        possible_bonds = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
        ]  # , Chem.rdchem.BondType.AROMATIC
        self.atom_type_num = len(possible_atoms)
        self.possible_atom_types = np.array(possible_atoms)
        self.possible_bond_types = np.array(possible_bonds, dtype=object)

        if self.has_feature:
            # self.d_n = len(self.possible_atom_types) + len(
            #     self.possible_formal_charge) + len(
            #     self.possible_implicit_valence) + len(self.possible_ring_atom) + \
            #       len(self.possible_degree) + len(self.possible_hybridization)
            self.d_n = len(self.possible_atom_types) + 6  # 6 is the ring feature
        else:
            self.d_n = len(self.possible_atom_types)

        self.max_action = max_action
        self.min_action = min_action
        if data_type == "gdb":
            self.max_atom = 13 + len(possible_atoms)  # gdb 13
        elif data_type == "zinc":
            if self.is_conditional:
                self.max_atom = 38 + len(possible_atoms) + self.min_action  # ZINC
            else:
                self.max_atom = 38 + len(possible_atoms)  # ZINC  + self.min_action

        self.logp_ratio = logp_ratio
        self.qed_ratio = qed_ratio
        self.sa_ratio = sa_ratio
        self.reward_step_total = reward_step_total
        self.action_space = gym.spaces.MultiDiscrete(
            [self.max_atom, self.max_atom, 3, 2]
        )
        self.observation_space = {}
        self.observation_space["adj"] = gym.Space(
            shape=[len(possible_bonds), self.max_atom, self.max_atom]
        )
        self.observation_space["node"] = gym.Space(shape=[1, self.max_atom, self.d_n])

        self.counter = 0

        ## load expert data
        cwd = os.path.dirname(__file__)
        if data_type == "gdb":
            path = os.path.join(
                os.path.dirname(cwd), "dataset", "gdb13.rand1M.smi.gz"
            )  # gdb 13
        elif data_type == "zinc":
            path = os.path.join(
                os.path.dirname(cwd), "dataset", "250k_rndm_zinc_drugs_clean_sorted.smi"
            )  # ZINC
        self.dataset = gdb_dataset(path)

        ## load scaffold data if necessary
        self.has_scaffold = has_scaffold
        if has_scaffold:
            self.scaffold = load_scaffold()
            self.max_scaffold = 6

        self.level = (
            0  # for curriculum learning, level starts with 0, and increase afterwards
        )

    def level_up(self):
        self.level += 1

    def seed(self, seed):
        np.random.seed(seed=seed)
        random.seed(seed)

    def normalize_adj(self, adj):
        degrees = np.sum(adj, axis=2)
        # print('degrees',degrees)
        D = np.zeros((adj.shape[0], adj.shape[1], adj.shape[2]))
        for i in range(D.shape[0]):
            D[i, :, :] = np.diag(np.power(degrees[i, :], -0.5))
        adj_normal = D @ adj @ D
        adj_normal[np.isnan(adj_normal)] = 0
        return adj_normal

    # TODO(Bowen): The top try, except clause allows error messages from step
    # to be printed when running run_molecules.py. For debugging only
    def step(self, action):
        """
        Perform a given action
        :param action:
        :param action_type:
        :return: reward of 1 if resulting molecule graph does not exceed valency,
        -1 if otherwise
        """
        ### init
        info = {}  # info we care about
        self.mol_old = copy.deepcopy(self.mol)  # keep old mol
        total_atoms = self.mol.GetNumAtoms()

        ### take action
        if action[0, 3] == 0 or self.counter < self.min_action:  # not stop
            stop = False
            if action[0, 1] >= total_atoms:
                self._add_atom(action[0, 1] - total_atoms)  # add new node
                action[0, 1] = total_atoms  # new node id
                self._add_bond(action)  # add new edge
            else:
                self._add_bond(action)  # add new edge
        else:  # stop
            stop = True

        ### calculate intermediate rewards
        if self.check_valency():
            if (
                self.mol.GetNumAtoms()
                + self.mol.GetNumBonds()
                - self.mol_old.GetNumAtoms()
                - self.mol_old.GetNumBonds()
                > 0
            ):
                reward_step = (
                    self.reward_step_total / self.max_atom
                )  # successfully add node/edge
                self.smile_list.append(self.get_final_smiles())
            else:
                reward_step = -self.reward_step_total / self.max_atom  # edge exist
        else:
            reward_step = -self.reward_step_total / self.max_atom  # invalid action
            self.mol = self.mol_old

        ### calculate terminal rewards
        # todo: add terminal action

        if self.is_conditional:
            terminate_condition = (
                self.mol.GetNumAtoms()
                >= self.max_atom - self.possible_atom_types.shape[0] - self.min_action
                or self.counter >= self.max_action
                or stop
            ) and self.counter >= self.min_action
        else:
            terminate_condition = (
                self.mol.GetNumAtoms()
                >= self.max_atom - self.possible_atom_types.shape[0]
                or self.counter >= self.max_action
                or stop
            ) and self.counter >= self.min_action
        if terminate_condition or self.force_final:
            # default reward
            reward_valid = 2
            reward_qed = 0
            reward_sa = 0
            reward_logp = 0
            reward_final = 0
            flag_steric_strain_filter = True
            flag_zinc_molecule_filter = True

            if not self.check_chemical_validity():
                reward_valid -= 5
            else:
                # final mol object where any radical electrons are changed to bonds to hydrogen
                final_mol = self.get_final_mol()
                s = Chem.MolToSmiles(final_mol, isomericSmiles=True)
                final_mol = Chem.MolFromSmiles(s)

                # mol filters with negative rewards
                if not steric_strain_filter(
                    final_mol
                ):  # passes 3D conversion, no excessive strain
                    reward_valid -= 1
                    flag_steric_strain_filter = False
                if not zinc_molecule_filter(
                    final_mol
                ):  # does not contain any problematic functional groups
                    reward_valid -= 1
                    flag_zinc_molecule_filter = False

                # global oracle_call
                # oracle_call += 1

                try:
                    print("smiles", Chem.MolToSmiles(final_mol))
                except:
                    print("invalid smiles")

                # property rewards
                # try:
                if True:
                    # 1. QED reward. Can have values [0, 1]. Higher the better
                    reward_qed += qed(final_mol) * self.qed_ratio
                    # 2. Synthetic accessibility reward. Values naively normalized to [0, 1]. Higher the better
                    sa = -1 * calculateScore(final_mol)
                    reward_sa += (sa + 10) / (10 - 1) * self.sa_ratio
                    # 3. Logp reward. Higher the better
                    # reward_logp += MolLogP(self.mol)/10 * self.logp_ratio
                    reward_logp += reward_penalized_log_p(final_mol) * self.logp_ratio
                    if self.reward_type == "logppen":
                        reward_final += reward_penalized_log_p(final_mol) / 3
                    elif self.reward_type == "logp_target":
                        # reward_final += reward_target(final_mol,target=self.reward_target,ratio=0.5,val_max=2,val_min=-2,func=MolLogP)
                        # reward_final += reward_target_logp(final_mol,target=self.reward_target)
                        reward_final += reward_target_new(
                            final_mol,
                            MolLogP,
                            x_start=self.reward_target,
                            x_mid=self.reward_target + 0.25,
                        )
                    elif self.reward_type == "qed":
                        reward_final += reward_qed * 2
                    elif self.reward_type == "qedsa":
                        reward_final += reward_qed * 1.5 + reward_sa * 0.5
                    elif self.reward_type == "qed_target":
                        # reward_final += reward_target(final_mol,target=self.reward_target,ratio=0.1,val_max=2,val_min=-2,func=qed)
                        reward_final += reward_target_qed(
                            final_mol, target=self.reward_target
                        )
                    elif self.reward_type == "mw_target":
                        # reward_final += reward_target(final_mol,target=self.reward_target,ratio=40,val_max=2,val_min=-2,func=rdMolDescriptors.CalcExactMolWt)
                        # reward_final += reward_target_mw(final_mol,target=self.reward_target)
                        reward_final += reward_target_new(
                            final_mol,
                            rdMolDescriptors.CalcExactMolWt,
                            x_start=self.reward_target,
                            x_mid=self.reward_target + 25,
                        )

                    elif self.reward_type == "docking":
                        try:
                            smiles0 = Chem.MolToSmiles(final_mol)
                            docking_score = oracle2(smiles0)
                        except:
                            docking_score = 0.0
                        if docking_score == None:
                            docking_score = 0.0
                        smiles2docking[smiles0] = docking_score
                        reward_final += -docking_score
                        oracle_call_cnt = get_oracle_num()
                        update_oracle_num()
                        oracle_call_cnt += 1
                        if oracle_call_cnt % 100 == 0:
                            smiles_score_lst = [
                                (smi, value) for smi, value in smiles2docking.items()
                            ]
                            smiles_score_lst.sort(key=lambda x: x[1])
                            with open(
                                "result/" + str(oracle_call_cnt) + ".txt", "w"
                            ) as fout:
                                for smi, score in smiles_score_lst[:100]:
                                    fout.write(smi + "\t" + str(score) + "\n")

                    elif self.reward_type == "gan":
                        reward_final = 0
                    else:
                        print("--------reward error!")
                        reward_final = 0

                    print("final reward", reward_final)

                # except: # if any property reward error, reset all
                #     print('================reward error')

            new = True  # end of episode
            if self.force_final:
                reward = reward_final
            else:
                reward = reward_step + reward_valid + reward_final
            info["smile"] = self.get_final_smiles()
            if self.is_conditional:
                info["reward_valid"] = self.conditional[-1]  ### temp change
            else:
                info["reward_valid"] = reward_valid
            info["reward_qed"] = reward_qed
            info["reward_sa"] = reward_sa
            info["final_stat"] = reward_final
            info["reward"] = reward
            info["flag_steric_strain_filter"] = flag_steric_strain_filter
            info["flag_zinc_molecule_filter"] = flag_zinc_molecule_filter
            info["stop"] = stop

        ### use stepwise reward
        else:
            new = False
            # print('counter', self.counter, 'new', new, 'reward_step', reward_step)
            reward = reward_step

        # get observation
        ob = self.get_observation()

        self.counter += 1
        if new:
            self.counter = 0

        return ob, reward, new, info

    def reset(self, smile=None):
        """
        to avoid error, assume an atom already exists
        :return: ob
        """
        if self.is_conditional:
            self.conditional = random.sample(self.conditional_list, 1)[0]
            self.mol = Chem.RWMol(Chem.MolFromSmiles(self.conditional[0]))
            Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        elif smile is not None:
            self.mol = Chem.RWMol(Chem.MolFromSmiles(smile))
            Chem.SanitizeMol(self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        else:
            self.mol = Chem.RWMol()
            # self._add_atom(np.random.randint(len(self.possible_atom_types)))  # random add one atom
            self._add_atom(0)  # always add carbon first
        self.smile_list = [self.get_final_smiles()]
        self.counter = 0
        ob = self.get_observation()
        return ob

    def render(self, mode="human", close=False):
        return

    def _add_atom(self, atom_type_id):
        """
        Adds an atom
        :param atom_type_id: atom_type id
        :return:
        """
        # assert action.shape == (len(self.possible_atom_types),)
        # atom_type_idx = np.argmax(action)
        atom_symbol = self.possible_atom_types[atom_type_id]
        self.mol.AddAtom(Chem.Atom(atom_symbol))

    def _add_bond(self, action):
        """

        :param action: [first_node, second_node, bong_type_id]
        :return:
        """
        # GetBondBetweenAtoms fails for np.int64
        bond_type = self.possible_bond_types[action[0, 2]]

        # if bond exists between current atom and other atom, modify the bond
        # type to new bond type. Otherwise create bond between current atom and
        # other atom with the new bond type
        bond = self.mol.GetBondBetweenAtoms(int(action[0, 0]), int(action[0, 1]))
        if bond:
            # print('bond exist!')
            return False
        else:
            self.mol.AddBond(int(action[0, 0]), int(action[0, 1]), order=bond_type)
            # bond = self.mol.GetBondBetweenAtoms(int(action[0, 0]), int(action[0, 1]))
            # bond.SetIntProp('ordering',self.mol.GetNumBonds())
            return True

    def _modify_bond(self, action):
        """
        Adds or modifies a bond (currently no deletion is allowed)
        :param action: np array of dim N-1 x d_e, where N is the current total
        number of atoms, d_e is the number of bond types
        :return:
        """
        assert action.shape == (self.current_atom_idx, len(self.possible_bond_types))
        other_atom_idx = int(np.argmax(action.sum(axis=1)))  # b/c
        # GetBondBetweenAtoms fails for np.int64
        bond_type_idx = np.argmax(action.sum(axis=0))
        bond_type = self.possible_bond_types[bond_type_idx]

        # if bond exists between current atom and other atom, modify the bond
        # type to new bond type. Otherwise create bond between current atom and
        # other atom with the new bond type
        bond = self.mol.GetBondBetweenAtoms(self.current_atom_idx, other_atom_idx)
        if bond:
            bond.SetBondType(bond_type)
        else:
            self.mol.AddBond(self.current_atom_idx, other_atom_idx, order=bond_type)
            self.total_bonds += 1

    def get_num_atoms(self):
        return self.total_atoms

    def get_num_bonds(self):
        return self.total_bonds

    def check_chemical_validity(self):
        """
        Checks the chemical validity of the mol object. Existing mol object is
        not modified. Radicals pass this test.
        :return: True if chemically valid, False otherwise
        """
        s = Chem.MolToSmiles(self.mol, isomericSmiles=True)
        m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
        if m:
            return True
        else:
            return False

    def check_valency(self):
        """
        Checks that no atoms in the mol have exceeded their possible
        valency
        :return: True if no valency issues, False otherwise
        """
        try:
            Chem.SanitizeMol(
                self.mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES
            )
            return True
        except ValueError:
            return False

    # TODO(Bowen): check if need to sanitize again
    def get_final_smiles(self):
        """
        Returns a SMILES of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = convert_radical_electrons_to_hydrogens(self.mol)
        return Chem.MolToSmiles(m, isomericSmiles=True)

    # TODO(Bowen): check if need to sanitize again
    def get_final_mol(self):
        """
        Returns a rdkit mol object of the final molecule. Converts any radical
        electrons into hydrogens. Works only if molecule is valid
        :return: SMILES
        """
        m = convert_radical_electrons_to_hydrogens(self.mol)
        return m

    def get_observation(self):
        """
        ob['adj']:d_e*n*n --- 'E'
        ob['node']:1*n*d_n --- 'F'
        n = atom_num + atom_type_num
        """
        mol = copy.deepcopy(self.mol)
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        n = mol.GetNumAtoms()
        n_shift = len(self.possible_atom_types)  # assume isolated nodes new nodes exist

        F = np.zeros((1, self.max_atom, self.d_n))
        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            atom_symbol = a.GetSymbol()
            if self.has_feature:
                formal_charge = a.GetFormalCharge()
                implicit_valence = a.GetImplicitValence()
                ring_atom = a.IsInRing()
                degree = a.GetDegree()
                hybridization = a.GetHybridization()
            # print(atom_symbol,formal_charge,implicit_valence,ring_atom,degree,hybridization)
            if self.has_feature:
                # float_array = np.concatenate([(atom_symbol ==
                #                                self.possible_atom_types),
                #                               (formal_charge ==
                #                                self.possible_formal_charge),
                #                               (implicit_valence ==
                #                                self.possible_implicit_valence),
                #                               (ring_atom ==
                #                                self.possible_ring_atom),
                #                               (degree == self.possible_degree),
                #                               (hybridization ==
                #                                self.possible_hybridization)]).astype(float)
                float_array = np.concatenate(
                    [
                        (atom_symbol == self.possible_atom_types),
                        ([not a.IsInRing()]),
                        ([a.IsInRingSize(3)]),
                        ([a.IsInRingSize(4)]),
                        ([a.IsInRingSize(5)]),
                        ([a.IsInRingSize(6)]),
                        (
                            [
                                a.IsInRing()
                                and (not a.IsInRingSize(3))
                                and (not a.IsInRingSize(4))
                                and (not a.IsInRingSize(5))
                                and (not a.IsInRingSize(6))
                            ]
                        ),
                    ]
                ).astype(float)
            else:
                float_array = (atom_symbol == self.possible_atom_types).astype(float)
            # assert float_array.sum() == 6   # because there are 6 types of one
            # print(float_array,float_array.sum())
            # hot atom features
            F[0, atom_idx, :] = float_array
        # add the atom features for the auxiliary atoms. We only include the
        # atom symbol features
        auxiliary_atom_features = np.zeros((n_shift, self.d_n))  # for padding
        temp = np.eye(n_shift)
        auxiliary_atom_features[: temp.shape[0], : temp.shape[1]] = temp
        F[0, n : n + n_shift, :] = auxiliary_atom_features
        # print('n',n,'n+n_shift',n+n_shift,auxiliary_atom_features.shape)

        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, self.max_atom, self.max_atom))
        for i in range(d_e):
            E[i, : n + n_shift, : n + n_shift] = np.eye(n + n_shift)
        for b in self.mol.GetBonds():  # self.mol, very important!! no aromatic
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)
            try:
                assert float_array.sum() != 0
            except:
                print("error", bond_type)
            E[:, begin_idx, end_idx] = float_array
            E[:, end_idx, begin_idx] = float_array
        ob = {}
        if self.is_normalize:
            E = self.normalize_adj(E)
        ob["adj"] = E
        ob["node"] = F
        return ob

    def get_observation_mol(self, mol):
        """
        ob['adj']:b*n*n --- 'E'
        ob['node']:1*n*m --- 'F'
        n = atom_num + atom_type_num
        """

        n = self.max_scaffold
        d_n = len(self.possible_atom_types)
        F = np.zeros((1, n, d_n))
        for a in mol.GetAtoms():
            atom_idx = a.GetIdx()
            atom_symbol = a.GetSymbol()
            float_array = (atom_symbol == self.possible_atom_types).astype(float)
            assert float_array.sum() != 0
            F[0, atom_idx, :] = float_array

        d_e = len(self.possible_bond_types)
        E = np.zeros((d_e, n, n))
        for i in range(d_e):
            E[i, :, :] = np.eye(n)
        for b in mol.GetBonds():
            begin_idx = b.GetBeginAtomIdx()
            end_idx = b.GetEndAtomIdx()
            bond_type = b.GetBondType()
            float_array = (bond_type == self.possible_bond_types).astype(float)
            assert float_array.sum() != 0
            E[:, begin_idx, end_idx] = float_array
            E[:, end_idx, begin_idx] = float_array
        ob = {}
        if self.is_normalize:
            E = self.normalize_adj(E)
        ob["adj"] = E
        ob["node"] = F
        return ob

    def get_observation_scaffold(self):
        ob = {}
        atom_type_num = len(self.possible_atom_types)
        bond_type_num = len(self.possible_bond_types)
        batch_size = len(self.scaffold)
        ob["node"] = np.zeros((batch_size, 1, self.max_scaffold, atom_type_num))
        ob["adj"] = np.zeros(
            (batch_size, bond_type_num, self.max_scaffold, self.max_scaffold)
        )
        for idx, mol in enumerate(self.scaffold):
            ob_temp = self.get_observation_mol(mol)
            ob["node"][idx] = ob_temp["node"]
            ob["adj"][idx] = ob_temp["adj"]
        return ob

    def get_expert(
        self, batch_size, is_final=False, curriculum=0, level_total=6, level=0
    ):
        ob = {}
        atom_type_num = len(self.possible_atom_types)
        bond_type_num = len(self.possible_bond_types)
        ob["node"] = np.zeros((batch_size, 1, self.max_atom, self.d_n))
        ob["adj"] = np.zeros((batch_size, bond_type_num, self.max_atom, self.max_atom))

        ac = np.zeros((batch_size, 4))
        ### select molecule
        dataset_len = len(self.dataset)
        for i in range(batch_size):
            is_final_temp = is_final
            # print('--------------------------------------------------')
            ### get a subgraph
            if curriculum == 1:
                ratio_start = level / float(level_total)
                ratio_end = (level + 1) / float(level_total)
                idx = np.random.randint(
                    int(ratio_start * dataset_len), int(ratio_end * dataset_len)
                )
            else:
                idx = np.random.randint(0, dataset_len)
            mol = self.dataset[idx]
            # print('ob_before',Chem.MolToSmiles(mol, isomericSmiles=True))
            # from rdkit.Chem import Draw
            # Draw.MolToFile(mol, 'ob_before'+str(i)+'.png')
            # mol = self.dataset[i] # sanitity check
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            graph = mol_to_nx(mol)
            edges = graph.edges()
            # # always involve is_final probability
            # if is_final==False and np.random.rand()<1.0/batch_size:
            #     is_final = True

            # select the edge num for the subgraph
            if is_final_temp:
                edges_sub_len = len(edges)
            else:
                # edges_sub_len = random.randint(1,len(edges))
                edges_sub_len = random.randint(1, len(edges) + 1)
                if edges_sub_len == len(edges) + 1:
                    edges_sub_len = len(edges)
                    is_final_temp = True
            edges_sub = random.sample(edges, k=edges_sub_len)
            graph_sub = nx.Graph(edges_sub)
            graph_sub = max(nx.connected_component_subgraphs(graph_sub), key=len)
            if (
                is_final_temp
            ):  # when the subgraph the whole molecule, the expert show stop sign
                node1 = random.randint(0, mol.GetNumAtoms() - 1)
                while True:
                    node2 = random.randint(0, mol.GetNumAtoms() + atom_type_num - 1)
                    if node2 != node1:
                        break
                edge_type = random.randint(0, bond_type_num - 1)
                ac[i, :] = [node1, node2, edge_type, 1]  # stop
            else:
                ### random pick an edge from the subgraph, then remove it
                edge_sample = random.sample(graph_sub.edges(), k=1)
                graph_sub.remove_edges_from(edge_sample)
                graph_sub = max(nx.connected_component_subgraphs(graph_sub), key=len)
                edge_sample = edge_sample[0]  # get value
                ### get action
                if (
                    edge_sample[0] in graph_sub.nodes()
                    and edge_sample[1] in graph_sub.nodes()
                ):
                    node1 = graph_sub.nodes().index(edge_sample[0])
                    node2 = graph_sub.nodes().index(edge_sample[1])
                elif edge_sample[0] in graph_sub.nodes():
                    node1 = graph_sub.nodes().index(edge_sample[0])
                    node2 = (
                        np.argmax(
                            graph.node[edge_sample[1]]["symbol"]
                            == self.possible_atom_types
                        )
                        + graph_sub.number_of_nodes()
                    )
                elif edge_sample[1] in graph_sub.nodes():
                    node1 = graph_sub.nodes().index(edge_sample[1])
                    node2 = (
                        np.argmax(
                            graph.node[edge_sample[0]]["symbol"]
                            == self.possible_atom_types
                        )
                        + graph_sub.number_of_nodes()
                    )
                else:
                    print("Expert policy error!")
                edge_type = np.argmax(
                    graph[edge_sample[0]][edge_sample[1]]["bond_type"]
                    == self.possible_bond_types
                )
                ac[i, :] = [node1, node2, edge_type, 0]  # don't stop
                # print('action',[node1,node2,edge_type,0])
            # print('action',ac)
            # plt.axis("off")
            # nx.draw_networkx(graph_sub)
            # plt.show()
            ### get observation
            # rw_mol = Chem.RWMol()
            n = graph_sub.number_of_nodes()
            for node_id, node in enumerate(graph_sub.nodes()):
                if self.has_feature:
                    # float_array = np.concatenate([(graph.node[node]['symbol'] ==
                    #                                self.possible_atom_types),
                    #                               (graph.node[node]['formal_charge'] ==
                    #                                self.possible_formal_charge),
                    #                               (graph.node[node]['implicit_valence'] ==
                    #                                self.possible_implicit_valence),
                    #                               (graph.node[node]['ring_atom'] ==
                    #                                self.possible_ring_atom),
                    #                               (graph.node[node]['degree'] == self.possible_degree),
                    #                               (graph.node[node]['hybridization'] ==
                    #                                self.possible_hybridization)]).astype(float)
                    cycle_info = nx.cycle_basis(graph_sub, node)
                    cycle_len_info = [len(cycle) for cycle in cycle_info]
                    # print(cycle_len_info)
                    float_array = np.concatenate(
                        [
                            (graph.node[node]["symbol"] == self.possible_atom_types),
                            ([len(cycle_info) == 0]),
                            ([3 in cycle_len_info]),
                            ([4 in cycle_len_info]),
                            ([5 in cycle_len_info]),
                            ([6 in cycle_len_info]),
                            (
                                [
                                    len(cycle_info) != 0
                                    and (not 3 in cycle_len_info)
                                    and (not 4 in cycle_len_info)
                                    and (not 5 in cycle_len_info)
                                    and (not 6 in cycle_len_info)
                                ]
                            ),
                        ]
                    ).astype(float)
                else:
                    float_array = (
                        graph.node[node]["symbol"] == self.possible_atom_types
                    ).astype(float)

                # assert float_array.sum() == 6
                ob["node"][i, 0, node_id, :] = float_array
                # print('node',node_id,graph.node[node]['symbol'])
                # atom = Chem.Atom(graph.node[node]['symbol'])
                # rw_mol.AddAtom(atom)
            auxiliary_atom_features = np.zeros((atom_type_num, self.d_n))  # for padding
            temp = np.eye(atom_type_num)
            auxiliary_atom_features[: temp.shape[0], : temp.shape[1]] = temp
            ob["node"][i, 0, n : n + atom_type_num, :] = auxiliary_atom_features

            for j in range(bond_type_num):
                ob["adj"][i, j, : n + atom_type_num, : n + atom_type_num] = np.eye(
                    n + atom_type_num
                )
            for edge in graph_sub.edges():
                begin_idx = graph_sub.nodes().index(edge[0])
                end_idx = graph_sub.nodes().index(edge[1])
                bond_type = graph[edge[0]][edge[1]]["bond_type"]
                float_array = (bond_type == self.possible_bond_types).astype(float)
                assert float_array.sum() != 0
                ob["adj"][i, :, begin_idx, end_idx] = float_array
                ob["adj"][i, :, end_idx, begin_idx] = float_array
                # print('edge',begin_idx,end_idx,bond_type)
                # rw_mol.AddBond(begin_idx, end_idx, order=bond_type)
            if self.is_normalize:
                ob["adj"][i] = self.normalize_adj(ob["adj"][i])
            # print('ob',Chem.MolToSmiles(rw_mol, isomericSmiles=True))
            # from rdkit.Chem import Draw
            # Draw.MolToFile(rw_mol, 'ob' + str(i) + '.png')

        return ob, ac


## below are for general graph generation env


def caveman_special(c=2, k=20, p_path=0.1, p_edge=0.3):
    p = p_path
    path_count = max(int(np.ceil(p * k)), 1)
    G = nx.caveman_graph(c, k)
    # remove 50% edges
    p = 1 - p_edge
    for (u, v) in list(G.edges()):
        if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
            G.remove_edge(u, v)
    # add path_count links
    for i in range(path_count):
        u = np.random.randint(0, k)
        v = np.random.randint(k, k * 2)
        G.add_edge(u, v)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G


class GraphEnv(gym.Env):
    """
    Environment for a general graph
    """

    def __init__(self):
        pass

    def init(self, reward_step_total=1, is_normalize=0, dataset="ba"):
        """
        own init function, since gym does not support passing argument
        """
        self.is_normalize = bool(is_normalize)
        self.graph = nx.Graph()
        self.reward_step_total = reward_step_total

        self.counter = 0

        ## load expert data
        if dataset == "caveman":
            self.dataset = []
            for i in range(2, 3):
                for j in range(6, 11):
                    for k in range(20):
                        self.dataset.append(
                            caveman_special(i, j, p_edge=0.8)
                        )  # default 0.8
            self.max_node = 25
            self.max_action = 150
        elif dataset == "grid":
            self.dataset = []
            for i in range(2, 5):
                for j in range(2, 6):
                    self.dataset.append(nx.grid_2d_graph(i, j))
            self.max_node = 25
            self.max_action = 100
        else:
            print("default dataset: barabasi")
            self.dataset = []
            for i in range(4, 21):
                for j in range(3, 4):
                    for k in range(10):
                        self.dataset.append(nx.barabasi_albert_graph(i, j))
            self.max_node = 25
            self.max_action = 150

        self.action_space = gym.spaces.MultiDiscrete(
            [self.max_node, self.max_node, 3, 2]
        )
        self.observation_space = {}
        self.observation_space["adj"] = gym.Space(
            shape=[1, self.max_node, self.max_node]
        )
        self.observation_space["node"] = gym.Space(shape=[1, self.max_node, 1])

        self.level = (
            0  # for curriculum learning, level starts with 0, and increase afterwards
        )

        # compatible with molecule env
        self.max_atom = self.max_node
        self.atom_type_num = 1

    def level_up(self):
        self.level += 1

    def normalize_adj(self, adj):
        degrees = np.sum(adj, axis=2)
        # print('degrees',degrees)
        D = np.zeros((adj.shape[0], adj.shape[1], adj.shape[2]))
        for i in range(D.shape[0]):
            D[i, :, :] = np.diag(np.power(degrees[i, :], -0.5))
        adj_normal = D @ adj @ D
        adj_normal[np.isnan(adj_normal)] = 0
        return adj_normal

    # TODO(Bowen): check
    def step(self, action):
        """

        :param action:
        :return:
        """
        ### init
        info = {}  # info we care about
        self.graph_old = copy.deepcopy(self.graph)
        total_nodes = self.graph.number_of_nodes()

        ### take action
        if action[0, 3] == 0:  # not stop
            stop = False
            if action[0, 1] >= total_nodes:
                self.graph.add_node(int(action[0, 1]))
                self._add_edge(action)
            else:
                self._add_edge(action)  # add new edge
        else:  # stop
            stop = True

        ### calculate intermediate rewards
        # todo: add neccessary rules for the task
        if (
            self.graph.number_of_nodes()
            + self.graph.number_of_edges()
            - self.graph_old.number_of_nodes()
            - self.graph_old.number_of_edges()
            > 0
        ):
            reward_step = self.reward_step_total / self.max_node
            # successfully added node/edge
        else:
            reward_step = -self.reward_step_total / self.max_node  # edge
            self.graph = self.graph_old
            # already exists

        ### calculate and use terminal reward
        if (
            self.graph.number_of_nodes() >= self.max_node - 1
            or self.counter >= self.max_action
            or stop
        ):

            # property rewards
            ## todo: add property reward
            reward_terminal = 1  # arbitrary choice

            new = True  # end of episode
            reward = reward_step + reward_terminal

            # print terminal graph information
            info["final_stat"] = reward_terminal
            info["reward"] = reward
            info["stop"] = stop
        ### use stepwise reward
        else:
            new = False
            reward = reward_step

        # get observation
        ob = self.get_observation()

        self.counter += 1
        if new:
            self.counter = 0

        return ob, reward, new, info

    def reset(self):
        """
        to avoid error, assume a node already exists
        :return: ob
        """
        self.graph.clear()
        self.graph.add_node(0)
        self.counter = 0
        ob = self.get_observation()
        return ob

    # TODO(Bowen): is this necessary
    def render(self, mode="human", close=False):
        return

    # TODO(Bowen): check
    def _add_node(self):
        """

        :param node_type_id:
        :return:
        """
        new_node_idx = self.graph.number_of_nodes()
        self.graph.add_node(new_node_idx)

    # TODO(Bowen): check
    def _add_edge(self, action):
        """

        :param action: [first_node, second_node, edge_type_id]
        :return:
        """

        if self.graph.has_edge(int(action[0, 0]), int(action[0, 1])) or int(
            action[0, 0]
        ) == int(action[0, 1]):
            return False
        else:
            self.graph.add_edge(int(action[0, 0]), int(action[0, 1]))
            return True

    def get_final_graph(self):
        return self.graph

    # TODO(Bowen): check [for featured graph]
    # def get_observation(self):
    #     """
    #
    #     :return: ob, where ob['adj'] is E with dim b x n x n and ob['node']
    #     is F with dim 1 x n x m. NB: n = node_num + node_type_num
    #     """
    #     n = self.graph.number_of_nodes()
    #     n_shift = len(self.possible_node_types)  # assume isolated nodes new nodes exist
    #
    #     d_n = len(self.possible_node_types)
    #     F = np.zeros((1, self.max_node, d_n))
    #
    #     for node in self.graph.nodes_iter(data=True):
    #         node_idx = node[0]
    #         node_type = node[1]['type']
    #         float_array = (node_type == self.possible_node_types).astype(float)
    #         assert float_array.sum() != 0
    #         F[0, node_idx, :] = float_array
    #     temp = F[0, n:n + n_shift, :]
    #     F[0, n:n + n_shift, :] = np.eye(n_shift)
    #
    #     d_e = len(self.possible_edge_types)
    #     E = np.zeros((d_e, self.max_node, self.max_node))
    #     for i in range(d_e):
    #         E[i, :n + n_shift, :n + n_shift] = np.eye(n + n_shift)
    #     for e in self.graph.edges_iter(data=True):
    #         begin_idx = e[0]
    #         end_idx = e[1]
    #         edge_type = e[2]['type']
    #         float_array = (edge_type == self.possible_edge_types).astype(float)
    #         assert float_array.sum() != 0
    #         E[:, begin_idx, end_idx] = float_array
    #         E[:, end_idx, begin_idx] = float_array
    #     ob = {}
    #     if self.is_normalize:
    #         E = self.normalize_adj(E)
    #     ob['adj'] = E
    #     ob['node'] = F
    #     return ob

    # for graphs without features
    def get_observation(self, feature="deg"):
        """

        :return: ob, where ob['adj'] is E with dim b x n x n and ob['node']
        is F with dim 1 x n x m. NB: n = node_num + node_type_num
        """
        n = self.graph.number_of_nodes()
        F = np.zeros((1, self.max_node, 1))
        F[0, : n + 1, 0] = 1

        E = np.zeros((1, self.max_node, self.max_node))
        E[0, :n, :n] = np.asarray(nx.to_numpy_matrix(self.graph))[np.newaxis, :, :]
        E[0, : n + 1, : n + 1] += np.eye(n + 1)

        ob = {}
        if self.is_normalize:
            E = self.normalize_adj(E)
        ob["adj"] = E
        ob["node"] = F
        return ob

    def get_expert(
        self, batch_size, is_final=False, curriculum=0, level_total=6, level=0
    ):
        ob = {}
        ob["node"] = np.zeros((batch_size, 1, self.max_node, 1))
        ob["adj"] = np.zeros((batch_size, 1, self.max_node, self.max_node))

        ac = np.zeros((batch_size, 4))
        ### select graph
        dataset_len = len(self.dataset)
        for i in range(batch_size):
            ### get a subgraph
            if curriculum == 1:
                ratio_start = level / float(level_total)
                ratio_end = (level + 1) / float(level_total)
                idx = np.random.randint(
                    int(ratio_start * dataset_len), int(ratio_end * dataset_len)
                )
            else:
                idx = np.random.randint(0, dataset_len)
            graph = self.dataset[idx]
            edges = graph.edges()
            # select the edge num for the subgraph
            if is_final:
                edges_sub_len = len(edges)
            else:
                edges_sub_len = random.randint(1, len(edges))
            edges_sub = random.sample(edges, k=edges_sub_len)
            graph_sub = nx.Graph(edges_sub)
            graph_sub = max(nx.connected_component_subgraphs(graph_sub), key=len)
            if is_final:  # when the subgraph the whole graph, the expert show
                # stop sign
                node1 = random.randint(0, graph.number_of_nodes() - 1)
                while True:
                    node2 = random.randint(0, graph.number_of_nodes())
                    if node2 != node1:
                        break
                edge_type = 0
                ac[i, :] = [node1, node2, edge_type, 1]  # stop
            else:
                ### random pick an edge from the subgraph, then remove it
                edge_sample = random.sample(graph_sub.edges(), k=1)
                graph_sub.remove_edges_from(edge_sample)
                graph_sub = max(nx.connected_component_subgraphs(graph_sub), key=len)
                edge_sample = edge_sample[0]  # get value
                ### get action
                if (
                    edge_sample[0] in graph_sub.nodes()
                    and edge_sample[1] in graph_sub.nodes()
                ):
                    node1 = graph_sub.nodes().index(edge_sample[0])
                    node2 = graph_sub.nodes().index(edge_sample[1])
                elif edge_sample[0] in graph_sub.nodes():
                    node1 = graph_sub.nodes().index(edge_sample[0])
                    node2 = graph_sub.number_of_nodes()
                elif edge_sample[1] in graph_sub.nodes():
                    node1 = graph_sub.nodes().index(edge_sample[1])
                    node2 = graph_sub.number_of_nodes()
                else:
                    print("Expert policy error!")
                edge_type = 0
                ac[i, :] = [node1, node2, edge_type, 0]  # don't stop
                # print('action',[node1,node2,edge_type,0])
            # print('action',ac)
            # plt.axis("off")
            # nx.draw_networkx(graph_sub)
            # plt.show()
            ### get observation
            n = graph_sub.number_of_nodes()
            F = np.zeros((1, self.max_node, 1))
            F[0, : n + 1, 0] = 1
            if self.is_normalize:
                ob["adj"][i] = self.normalize_adj(F)
            else:
                ob["node"][i] = F
            # print(F)
            E = np.zeros((1, self.max_node, self.max_node))
            E[0, :n, :n] = np.asarray(nx.to_numpy_matrix(graph_sub))[np.newaxis, :, :]
            E[0, : n + 1, : n + 1] += np.eye(n + 1)
            ob["adj"][i] = E
            # print(E)

        return ob, ac


### YES/NO filters ###
def zinc_molecule_filter(mol):
    """
    Flags molecules based on problematic functional groups as
    provided set of ZINC rules from
    http://blaster.docking.org/filtering/rules_default.txt.
    :param mol: rdkit mol object
    :return: Returns True if molecule is okay (ie does not match any of
    therules), False if otherwise
    """
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.ZINC)
    catalog = FilterCatalog(params)
    return not catalog.HasMatch(mol)


# TODO(Bowen): check
def steric_strain_filter(mol, cutoff=0.82, max_attempts_embed=20, max_num_iters=200):
    """
    Flags molecules based on a steric energy cutoff after max_num_iters
    iterations of MMFF94 forcefield minimization. Cutoff is based on average
    angle bend strain energy of molecule
    :param mol: rdkit mol object
    :param cutoff: kcal/mol per angle . If minimized energy is above this
    threshold, then molecule fails the steric strain filter
    :param max_attempts_embed: number of attempts to generate initial 3d
    coordinates
    :param max_num_iters: number of iterations of forcefield minimization
    :return: True if molecule could be successfully minimized, and resulting
    energy is below cutoff, otherwise False
    """
    # check for the trivial cases of a single atom or only 2 atoms, in which
    # case there is no angle bend strain energy (as there are no angles!)
    if mol.GetNumAtoms() <= 2:
        return True

    # make copy of input mol and add hydrogens
    m = copy.deepcopy(mol)
    m_h = Chem.AddHs(m)

    # generate an initial 3d conformer
    try:
        flag = AllChem.EmbedMolecule(m_h, maxAttempts=max_attempts_embed)
        if flag == -1:
            # print("Unable to generate 3d conformer")
            return False
    except:  # to catch error caused by molecules such as C=[SH]1=C2OC21ON(N)OC(=O)NO
        # print("Unable to generate 3d conformer")
        return False

    # set up the forcefield
    AllChem.MMFFSanitizeMolecule(m_h)
    if AllChem.MMFFHasAllMoleculeParams(m_h):
        mmff_props = AllChem.MMFFGetMoleculeProperties(m_h)
        try:  # to deal with molecules such as CNN1NS23(=C4C5=C2C(=C53)N4Cl)S1
            ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
        except:
            # print("Unable to get forcefield or sanitization error")
            return False
    else:
        # print("Unrecognized atom type")
        return False

    # minimize steric energy
    try:
        ff.Minimize(maxIts=max_num_iters)
    except:
        # print("Minimization error")
        return False

    # ### debug ###
    # min_e = ff.CalcEnergy()
    # print("Minimized energy: {}".format(min_e))
    # ### debug ###

    # get the angle bend term contribution to the total molecule strain energy
    mmff_props.SetMMFFBondTerm(False)
    mmff_props.SetMMFFAngleTerm(True)
    mmff_props.SetMMFFStretchBendTerm(False)
    mmff_props.SetMMFFOopTerm(False)
    mmff_props.SetMMFFTorsionTerm(False)
    mmff_props.SetMMFFVdWTerm(False)
    mmff_props.SetMMFFEleTerm(False)

    ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)

    min_angle_e = ff.CalcEnergy()
    # print("Minimized angle bend energy: {}".format(min_angle_e))

    # find number of angles in molecule
    # TODO(Bowen): there must be a better way to get a list of all angles
    # from molecule... This is too hacky
    num_atoms = m_h.GetNumAtoms()
    atom_indices = range(num_atoms)
    angle_atom_triplets = itertools.permutations(atom_indices, 3)  # get all
    # possible 3 atom indices groups. Currently, each angle is represented by
    #  2 duplicate groups. Should remove duplicates here to be more efficient
    double_num_angles = 0
    for triplet in list(angle_atom_triplets):
        if mmff_props.GetMMFFAngleBendParams(m_h, *triplet):
            double_num_angles += 1
    num_angles = double_num_angles / 2  # account for duplicate angles

    # print("Number of angles: {}".format(num_angles))

    avr_angle_e = min_angle_e / num_angles

    # print("Average minimized angle bend energy: {}".format(avr_angle_e))

    # ### debug ###
    # for i in range(7):
    #     termList = [['BondStretch', False], ['AngleBend', False],
    #                 ['StretchBend', False], ['OopBend', False],
    #                 ['Torsion', False],
    #                 ['VdW', False], ['Electrostatic', False]]
    #     termList[i][1] = True
    #     mmff_props.SetMMFFBondTerm(termList[0][1])
    #     mmff_props.SetMMFFAngleTerm(termList[1][1])
    #     mmff_props.SetMMFFStretchBendTerm(termList[2][1])
    #     mmff_props.SetMMFFOopTerm(termList[3][1])
    #     mmff_props.SetMMFFTorsionTerm(termList[4][1])
    #     mmff_props.SetMMFFVdWTerm(termList[5][1])
    #     mmff_props.SetMMFFEleTerm(termList[6][1])
    #     ff = AllChem.MMFFGetMoleculeForceField(m_h, mmff_props)
    #     print('{0:>16s} energy: {1:12.4f} kcal/mol'.format(termList[i][0],
    #                                                  ff.CalcEnergy()))
    # ## end debug ###

    if avr_angle_e < cutoff:
        return True
    else:
        return False


### TARGET VALUE REWARDS ###


def reward_target(mol, target, ratio, val_max, val_min, func):
    x = func(mol)
    reward = max(-1 * np.abs((x - target) / ratio) + val_max, val_min)
    return reward


def reward_target_new(
    mol, func, r_max1=4, r_max2=2.25, r_mid=2, r_min=-2, x_start=500, x_mid=525
):
    x = func(mol)
    return max(
        (r_max1 - r_mid) / (x_start - x_mid) * np.abs(x - x_mid) + r_max1,
        (r_max2 - r_mid) / (x_start - x_mid) * np.abs(x - x_mid) + r_max2,
        r_min,
    )


def reward_target_logp(mol, target, ratio=0.5, max=4):
    """
    Reward for a target log p
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = MolLogP(mol)
    reward = -1 * np.abs((x - target) / ratio) + max
    return reward


def reward_target_penalizelogp(mol, target, ratio=3, max=4):
    """
    Reward for a target log p
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = reward_penalized_log_p(mol)
    reward = -1 * np.abs((x - target) / ratio) + max
    return reward


def reward_target_qed(mol, target, ratio=0.1, max=4):
    """
    Reward for a target log p
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = qed(mol)
    reward = -1 * np.abs((x - target) / ratio) + max
    return reward


def reward_target_mw(mol, target, ratio=40, max=4):
    """
    Reward for a target molecular weight
    :param mol: rdkit mol object
    :param target: float
    :return: float (-inf, max]
    """
    x = rdMolDescriptors.CalcExactMolWt(mol)
    reward = -1 * np.abs((x - target) / ratio) + max
    return reward


# TODO(Bowen): num rings is a discrete variable, so what is the best way to
# calculate the reward?
def reward_target_num_rings(mol, target):
    """
    Reward for a target number of rings
    :param mol: rdkit mol object
    :param target: int
    :return: float (-inf, 1]
    """
    x = rdMolDescriptors.CalcNumRings(mol)
    reward = -1 * (x - target) ** 2 + 1
    return reward


# TODO(Bowen): more efficient if we precalculate the target fingerprint
from rdkit import DataStructs


def reward_target_molecule_similarity(
    mol, target, radius=2, nBits=2048, useChirality=True
):
    """
    Reward for a target molecule similarity, based on tanimoto similarity
    between the ECFP fingerprints of the x molecule and target molecule
    :param mol: rdkit mol object
    :param target: rdkit mol object
    :return: float, [0.0, 1.0]
    """
    x = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=nBits, useChirality=useChirality
    )
    target = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        target, radius=radius, nBits=nBits, useChirality=useChirality
    )
    return DataStructs.TanimotoSimilarity(x, target)


### TERMINAL VALUE REWARDS ###


def reward_penalized_log_p(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = MolLogP(mol)
    SA = -calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    return normalized_log_p + normalized_SA + normalized_cycle


# # TEST compare with junction tree paper examples from Figure 7
# assert round(reward_penalized_log_p(Chem.MolFromSmiles('ClC1=CC=C2C(C=C(C('
#                                                        'C)=O)C(C(NC3=CC(NC('
#                                                        'NC4=CC(C5=C('
#                                                        'C)C=CC=C5)=CC=C4)=O)=CC=C3)=O)=C2)=C1')), 2) == 5.30
# assert round(reward_penalized_log_p(Chem.MolFromSmiles('CC(NC1=CC(C2=CC=CC('
#                                                        'NC(NC3=CC=CC(C4=CC('
#                                                        'F)=CC=C4)=C3)=O)=C2)=CC=C1)=O')), 2) == 4.49
# assert round(reward_penalized_log_p(Chem.MolFromSmiles('ClC(C('
#                                                        'Cl)=C1)=CC=C1NC2=CC=CC=C2C(NC(NC3=C(C(NC4=C(Cl)C=CC=C4)=S)C=CC=C3)=O)=O')), 2) == 4.93


def get_normalized_values():
    fname = "/home/bowen/pycharm_deployment_directory/rl_graph_generation/gym-molecule/gym_molecule/dataset/250k_rndm_zinc_drugs_clean.smi"
    with open(fname) as f:
        smiles = f.readlines()

    for i in range(len(smiles)):
        smiles[i] = smiles[i].strip()
    smiles_rdkit = []

    for i in range(len(smiles)):
        smiles_rdkit.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles[i])))
    print(i)

    logP_values = []
    for i in range(len(smiles)):
        logP_values.append(MolLogP(Chem.MolFromSmiles(smiles_rdkit[i])))
    print(i)

    SA_scores = []
    for i in range(len(smiles)):
        SA_scores.append(-calculateScore(Chem.MolFromSmiles(smiles_rdkit[i])))
    print(i)

    cycle_scores = []
    for i in range(len(smiles)):
        cycle_list = nx.cycle_basis(
            nx.Graph(
                Chem.rdmolops.GetAdjacencyMatrix(Chem.MolFromSmiles(smiles_rdkit[i]))
            )
        )
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_scores.append(-cycle_length)
    print(i)

    SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(
        SA_scores
    )
    logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(
        logP_values
    )
    cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(
        cycle_scores
    )

    return (
        np.mean(SA_scores),
        np.std(SA_scores),
        np.mean(logP_values),
        np.std(logP_values),
        np.mean(cycle_scores),
        np.std(cycle_scores),
    )


# smile = 'C'*38
smile = "CCCCCCCCCC(CCC)(CCCCCCC)CCCCCCCCC(CCCCC)CC(C)C"
print(smile, reward_penalized_log_p(Chem.MolFromSmiles(smile)))

if __name__ == "__main__":
    env = gym.make("molecule-v0")  # in gym format
    # env = GraphEnv()
    # env.init(has_scaffold=True)

    ## debug
    m_env = MoleculeEnv()
    m_env.init(data_type="zinc", has_feature=True, is_conditional=True)


######### GraphEnv potentially with feature
# class GraphEnv(gym.Env):
#     """
#     Environment for a general graph
#     """
#     def __init__(self):
#         pass
#     # TODO(Bowen): check
#     def init(self, data_type='simple', reward_step_total=1, is_normalize=0):
#         '''
#         own init function, since gym does not support passing argument
#         '''
#         self.is_normalize = bool(is_normalize)
#         self.graph = nx.Graph()
#         if data_type == 'simple':
#             possible_nodes = ['n']
#             possible_edges = ['e']
#             self.max_node = 100 + len(possible_nodes)
#         else:
#             raise ValueError('Invalid data type')
#         self.node_type_num = len(possible_nodes)
#         self.possible_node_types = np.array(possible_nodes)
#         self.possible_edge_types = np.array(possible_edges)
#
#         self.max_action = 128
#         self.reward_step_total = reward_step_total
#         self.action_space = gym.spaces.MultiDiscrete([self.max_node, self.max_node, 3, 2])
#         self.observation_space = {}
#         self.observation_space['adj'] = gym.Space(shape=[len(possible_edges),
#                                                          self.max_node,
#                                                          self.max_node])
#         self.observation_space['node'] = gym.Space(shape=[1, self.max_node,
#                                                           len(possible_nodes)])
#
#         self.counter = 0
#
#         ## load expert data
#         cwd = os.path.dirname(__file__)
#         # if data_type == 'simple':
#         #     path = os.path.join(os.path.dirname(cwd), 'dataset',
#         #                         'simple')  #TODO: Add a suitable dataset
#         # self.dataset = dataset(path)    #TODO: Add a suitable dataset loader
#         if data_type == 'simple':
#             self.dataset = [nx.barabasi_albert_graph(100,2) for i in range(200)]
#
#
#         self.level = 0  # for curriculum learning, level starts with 0, and increase afterwards
#
#     def level_up(self):
#         self.level += 1
#
#     def normalize_adj(self, adj):
#         degrees = np.sum(adj, axis=2)
#         # print('degrees',degrees)
#         D = np.zeros((adj.shape[0], adj.shape[1], adj.shape[2]))
#         for i in range(D.shape[0]):
#             D[i, :, :] = np.diag(np.power(degrees[i, :], -0.5))
#         adj_normal = D @ adj @ D
#         adj_normal[np.isnan(adj_normal)] = 0
#         return adj_normal
#
#     # TODO(Bowen): check
#     def step(self, action):
#         """
#
#         :param action:
#         :return:
#         """
#         ### init
#         info = {}  # info we care about
#         self.graph_old = copy.deepcopy(self.graph)
#         total_nodes = self.graph.number_of_nodes()
#
#         ### take action
#         if action[0, 3] == 0:   # not stop
#             stop = False
#             if action[0, 1] >= total_nodes:
#                 self._add_node(action[0, 1] - total_nodes)  # add new node
#                 action[0, 1] = total_nodes  # new node id
#                 self._add_edge(action)  # add new edge
#             else:
#                 self._add_edge(action)  # add new edge
#         else:   # stop
#             stop = True
#
#         ### calculate intermediate rewards
#         if self.graph.number_of_nodes() + self.graph.number_of_edges() - \
#             self.graph_old.number_of_nodes() - \
#             self.graph_old.number_of_edges() > 0:
#             reward_step = self.reward_step_total / self.max_node
#             # successfully added node/edge
#         else:
#             reward_step = -self.reward_step_total / self.max_node # edge
#             # already exists
#
#         ### calculate and use terminal reward
#         if self.graph.number_of_nodes() >= self.max_node - \
#             self.possible_node_types.shape[0] or self.counter >= \
#             self.max_action or stop:
#
#             # property rewards
#             reward_terminal = 1 # arbitrary choice
#
#             new = True  # end of episode
#             reward = reward_step + reward_terminal
#
#             # print terminal graph information
#             info['reward_terminal'] = reward_terminal
#             info['reward'] = reward
#             info['stop'] = stop
#         ### use stepwise reward
#         else:
#             new = False
#             reward = reward_step
#
#         # get observation
#         ob = self.get_observation()
#
#         self.counter += 1
#         if new:
#             self.counter = 0
#
#         return ob, reward, new, info
#
#     def reset(self):
#         """
#         to avoid error, assume a node already exists
#         :return: ob
#         """
#         self.graph.clear()
#         self._add_node(0)
#         self.counter = 0
#         ob = self.get_observation()
#         return ob
#
#     # TODO(Bowen): is this necessary
#     def render(self, mode='human', close=False):
#         return
#
#     # TODO(Bowen): check
#     def _add_node(self, node_type_id):
#         """
#
#         :param node_type_id:
#         :return:
#         """
#         new_node_idx = self.graph.number_of_nodes()
#         self.graph.add_node(new_node_idx, type=self.possible_node_types[node_type_id])
#
#     # TODO(Bowen): check
#     def _add_edge(self, action):
#         """
#
#         :param action: [first_node, second_node, edge_type_id]
#         :return:
#         """
#         edge_type = self.possible_edge_types[action[0, 2]]
#
#         if self.graph.has_edge(int(action[0,0]), int(action[0,1])):
#             return False
#         else:
#             self.graph.add_edge(int(action[0,0]), int(action[0,1]), type=edge_type)
#             return True
#
#     def get_final_graph(self):
#         return self.graph
#
#     # TODO(Bowen): check [for featured graph]
#     # def get_observation(self):
#     #     """
#     #
#     #     :return: ob, where ob['adj'] is E with dim b x n x n and ob['node']
#     #     is F with dim 1 x n x m. NB: n = node_num + node_type_num
#     #     """
#     #     n = self.graph.number_of_nodes()
#     #     n_shift = len(self.possible_node_types)  # assume isolated nodes new nodes exist
#     #
#     #     d_n = len(self.possible_node_types)
#     #     F = np.zeros((1, self.max_node, d_n))
#     #
#     #     for node in self.graph.nodes_iter(data=True):
#     #         node_idx = node[0]
#     #         node_type = node[1]['type']
#     #         float_array = (node_type == self.possible_node_types).astype(float)
#     #         assert float_array.sum() != 0
#     #         F[0, node_idx, :] = float_array
#     #     temp = F[0, n:n + n_shift, :]
#     #     F[0, n:n + n_shift, :] = np.eye(n_shift)
#     #
#     #     d_e = len(self.possible_edge_types)
#     #     E = np.zeros((d_e, self.max_node, self.max_node))
#     #     for i in range(d_e):
#     #         E[i, :n + n_shift, :n + n_shift] = np.eye(n + n_shift)
#     #     for e in self.graph.edges_iter(data=True):
#     #         begin_idx = e[0]
#     #         end_idx = e[1]
#     #         edge_type = e[2]['type']
#     #         float_array = (edge_type == self.possible_edge_types).astype(float)
#     #         assert float_array.sum() != 0
#     #         E[:, begin_idx, end_idx] = float_array
#     #         E[:, end_idx, begin_idx] = float_array
#     #     ob = {}
#     #     if self.is_normalize:
#     #         E = self.normalize_adj(E)
#     #     ob['adj'] = E
#     #     ob['node'] = F
#     #     return ob
#
#
#     # for graphs without features
#     def get_observation(self,feature='deg'):
#         """
#
#         :return: ob, where ob['adj'] is E with dim b x n x n and ob['node']
#         is F with dim 1 x n x m. NB: n = node_num + node_type_num
#         """
#         n = self.graph.number_of_nodes()
#         F = np.zeros((1, self.max_node, 1))
#         F[0,:n+1,0] = 1
#
#         E = np.zeros((1, self.max_node, self.max_node))
#         E[0,:n,:n] = np.asarray(nx.to_numpy_matrix(self.graph))[np.newaxis,:,:]
#         E[0,:n+1,:n+1] += np.eye(n+1)
#
#         ob = {}
#         if self.is_normalize:
#             E = self.normalize_adj(E)
#         ob['adj'] = E
#         ob['node'] = F
#         return ob
#
#     def get_expert(self, batch_size, is_final=False, curriculum=0,
#                    level_total=6, level=0):
#         ob = {}
#         node_type_num = len(self.possible_node_types)
#         edge_type_num = len(self.possible_edge_types)
#         ob['node'] = np.zeros((batch_size, 1, self.max_node, node_type_num))
#         ob['adj'] = np.zeros(
#             (batch_size, edge_type_num, self.max_node, self.max_node))
#
#         ac = np.zeros((batch_size, 4))
#         ### select graph
#         dataset_len = len(self.dataset)
#         for i in range(batch_size):
#             # print('--------------------------------------------------')
#             ### get a subgraph
#             if curriculum == 1:
#                 ratio_start = level / float(level_total)
#                 ratio_end = (level + 1) / float(level_total)
#                 idx = np.random.randint(int(ratio_start * dataset_len),
#                                         int(ratio_end * dataset_len))
#             else:
#                 idx = np.random.randint(0, dataset_len)
#             graph = self.dataset[idx]
#             edges = graph.edges()
#             # select the edge num for the subgraph
#             if is_final:
#                 edges_sub_len = len(edges)
#             else:
#                 edges_sub_len = random.randint(1, len(edges))
#             edges_sub = random.sample(edges, k=edges_sub_len)
#             graph_sub = nx.Graph(edges_sub)
#             graph_sub = max(nx.connected_component_subgraphs(graph_sub),
#                             key=len)
#             if is_final:  # when the subgraph the whole graph, the expert show
#                 # stop sign
#                 node1 = random.randint(0, graph.number_of_nodes() - 1)
#                 while True:
#                     node2 = random.randint(0,
#                                            graph.number_of_nodes() + node_type_num - 1)
#                     if node2 != node1:
#                         break
#                 edge_type = random.randint(0, edge_type_num - 1)
#                 ac[i, :] = [node1, node2, edge_type, 1]  # stop
#             else:
#                 ### random pick an edge from the subgraph, then remove it
#                 edge_sample = random.sample(graph_sub.edges(), k=1)
#                 graph_sub.remove_edges_from(edge_sample)
#                 graph_sub = max(nx.connected_component_subgraphs(graph_sub),
#                                 key=len)
#                 edge_sample = edge_sample[0]  # get value
#                 ### get action
#                 if edge_sample[0] in graph_sub.nodes() and edge_sample[
#                     1] in graph_sub.nodes():
#                     node1 = graph_sub.nodes().index(edge_sample[0])
#                     node2 = graph_sub.nodes().index(edge_sample[1])
#                 elif edge_sample[0] in graph_sub.nodes():
#                     node1 = graph_sub.nodes().index(edge_sample[0])
#                     node2 = np.argmax(
#                         graph.node[edge_sample[1]][
#                             'type'] == self.possible_node_types) + \
#                             graph_sub.number_of_nodes()
#                 elif edge_sample[1] in graph_sub.nodes():
#                     node1 = graph_sub.nodes().index(edge_sample[1])
#                     node2 = np.argmax(
#                         graph.node[edge_sample[0]][
#                             'type'] == self.possible_node_types) + graph_sub.number_of_nodes()
#                 else:
#                     print('Expert policy error!')
#                 edge_type = np.argmax(graph[edge_sample[0]][edge_sample[1]][
#                                           'type'] == self.possible_edge_types)
#                 ac[i, :] = [node1, node2, edge_type, 0]  # don't stop
#                 # print('action',[node1,node2,edge_type,0])
#             # print('action',ac)
#             # plt.axis("off")
#             # nx.draw_networkx(graph_sub)
#             # plt.show()
#             ### get observation
#             n = graph_sub.number_of_nodes()
#             for node_id, node in enumerate(graph_sub.nodes()):
#                 float_array = (
#                 graph.node[node]['type'] == self.possible_node_types).astype(
#                     float)
#                 assert float_array.sum() != 0
#                 ob['node'][i, 0, node_id, :] = float_array
#             ob['node'][i, 0, n:n + node_type_num, :] = np.eye(node_type_num)
#
#             for j in range(edge_type_num):
#                 ob['adj'][i, j, :n + node_type_num,
#                 :n + node_type_num] = np.eye(n + node_type_num)
#             for edge in graph_sub.edges():
#                 begin_idx = graph_sub.nodes().index(edge[0])
#                 end_idx = graph_sub.nodes().index(edge[1])
#                 edge_type = graph[edge[0]][edge[1]]['type']
#                 float_array = (edge_type == self.possible_edge_types).astype(
#                     float)
#                 assert float_array.sum() != 0
#                 ob['adj'][i, :, begin_idx, end_idx] = float_array
#                 ob['adj'][i, :, end_idx, begin_idx] = float_array
#             if self.is_normalize:
#                 ob['adj'][i] = self.normalize_adj(ob['adj'][i])
#
#         return ob, ac
